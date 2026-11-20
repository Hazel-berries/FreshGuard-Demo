/**
 * MCTS with Importance Sampling for perishable inventory inference.
 * See mcts.py for the full theoretical rationale.
 *
 * Memory & performance optimisations vs the original JS port
 * ----------------------------------------------------------
 * weightedSurvivalSum : Float64Array(numIds)  instead of plain {}
 *   → ~700 B → ~120 B per node; eliminates string-key overhead and
 *     enables tight typed-array loops in backprop and rollout.
 *
 * maxNodes : hard cap on tree size
 *   → once reached, _expand is skipped; all remaining iterations do
 *     selection + rollout only, bounding memory regardless of n_iter.
 *
 * _rollout : reusable Float64Array buffer (this._survivalBuf)
 *   → no per-rollout allocation; direct state scan replaces Set+flat().
 *
 * _select : cache Math.log(parent.visits) per level; loop instead of reduce.
 *
 * applySell / applyCompleteShuffle (model.js) : no spread operators,
 *   no temporary neighbors array, no flat(), geometric sampling without
 *   intermediate weight array.
 */

import {
    applyCompleteShuffle,
    applySell,
    applyTransition,
    createRNG,
    totalItems,
} from './model.js';

const SELL = 'sell';
const COMPLETE_SHUFFLE = 'complete_shuffle';

// ---------------------------------------------------------------------------
// Tree node
// ---------------------------------------------------------------------------

class MCTSNode {
    /**
     * @param {number} numIds  Max item ID + 1 — sizes the typed survival array.
     */
    constructor(state, sellCount, logW, eventType, parent, numIds) {
        this.state     = state;
        this.sellCount = sellCount;
        this.logW      = logW;
        this.eventType = eventType;
        this.parent    = parent;
        this.children  = [];

        this.visits    = 0;
        this.weightSum = 0.0;
        // Float64Array: ~120 B fixed vs ~700 B for a plain JS object with 15 keys
        this.weightedSurvivalSum = new Float64Array(numIds);
    }

    isTerminal(nSells) { return this.sellCount >= nSells; }
}

// ---------------------------------------------------------------------------
// MCTS engine
// ---------------------------------------------------------------------------

export class InventoryMCTS {
    /**
     * @param {Array<Array<number>>} initialState
     * @param {object} params   { q, p_s, p_r, p_sell, n_end }
     * @param {object} options
     * @param {number} [options.csProposal]
     * @param {number} [options.maxChildren=1000]
     * @param {number} [options.eagerFraction=0.75]  Expand unconditionally until k < fraction*maxChildren; then opportunistic UCB.
     * @param {number} [options.maxDepth=100]
     * @param {number} [options.maxNodes=100000]  Hard cap on total tree nodes.
     * @param {number} [options.cExplore=√2]
     * @param {number} [options.nRollouts=1]
     * @param {number|null} [options.seed=null]
     */
    constructor(initialState, params, {
        csProposal       = null,
        maxChildren      = 1000,
        eagerFraction    = 0.75,   // expand unconditionally until k < eagerFraction*maxChildren
        maxDepth         = 100,
        maxNodes         = 100_000,
        cExplore         = Math.SQRT2,
        nRollouts        = 1,
        seed             = null,
    } = {}) {
        this.params      = params;
        this.cExplore    = cExplore;
        this.maxChildren = maxChildren;
        this.eagerCap    = Math.floor(eagerFraction * maxChildren); // exclusive threshold
        this.maxDepth    = maxDepth;
        this.maxNodes    = maxNodes;
        this.nRollouts   = nRollouts;
        this.rng         = createRNG(seed ?? Date.now());

        const nStart = totalItems(initialState);
        this.nSells = nStart - params.n_end;
        if (this.nSells < 0) throw new Error(
            `n_end (${params.n_end}) exceeds initial item count (${nStart}).`);

        // Proposal probabilities
        const pCS  = 1.0 - params.p_sell;
        this.qCS   = csProposal !== null ? csProposal : Math.max(pCS, 0.2);
        if (this.qCS < pCS) throw new Error(
            `csProposal (${this.qCS.toFixed(3)}) < 1-p_sell (${pCS.toFixed(3)})`);
        this.qSell = 1.0 - this.qCS;

        this.logWSell = this.qSell > 0
            ? Math.log(params.p_sell) - Math.log(this.qSell) : 0.0;
        this.logWCS   = (pCS > 0 && this.qCS > 0)
            ? Math.log(pCS) - Math.log(this.qCS) : -Infinity;

        // Compute numIds = maxItemId + 1 for typed-array sizing
        let maxId = 0;
        for (const col of initialState) for (const id of col) if (id > maxId) maxId = id;
        this._numIds = maxId + 1;
        this._nWords = Math.ceil(this._numIds / 32);

        // allIds as a sorted array (faster iteration than Set in hot loops)
        this.allIds       = new Set(initialState.flat());
        this._allIdsArray = [...this.allIds].sort((a, b) => a - b);

        // Reusable rollout accumulator — reset with fill(0) before each use
        this._survivalBuf = new Float64Array(this._numIds);
        // Reusable bitmask buffer for survivor-set keys — reset with fill(0) before each use
        this._keyBuf      = new Uint32Array(this._nWords);

        // Joint posterior over end-of-day survivor sets:
        //   key = BigInt bitmask of surviving item IDs
        //   val = cumulative IS weight across all rollouts that landed at that set
        // Sparse — only sets actually visited by rollouts appear.
        this.rolloutMap = new Map();

        this._nodeCount = 1;   // root
        this.root = new MCTSNode(initialState, 0, 0.0, null, null, this._numIds);
    }

    // ------------------------------------------------------------------
    // Public interface
    // ------------------------------------------------------------------

    /**
     * Run MCTS for nIterations.
     * @param {number} nIterations
     * @param {{ onProgress?: (done:number, total:number)=>void }} options
     */
    run(nIterations, { onProgress = null } = {}) {
        const reportEvery = Math.max(1, Math.floor(nIterations / 100));
        for (let i = 0; i < nIterations; i++) {
            this._iterate();
            if (onProgress && (i + 1) % reportEvery === 0) onProgress(i + 1, nIterations);
        }
        if (onProgress) onProgress(nIterations, nIterations);
    }

    /** Self-normalised IS estimate of P(item_i survives). */
    survivalProbabilities() {
        const result = new Map();
        const ws  = this.root.weightSum;
        const wss = this.root.weightedSurvivalSum;
        for (const id of this._allIdsArray) {
            result.set(id, ws > 0 ? wss[id] / ws : 0.0);
        }
        return result;
    }

    /** Expected number of items (optionally filtered) surviving to end-of-day. */
    expectedOnShelf(itemIds = null) {
        const ids  = itemIds ?? this.allIds;
        const probs = this.survivalProbabilities();
        let total = 0;
        for (const id of ids) total += probs.get(id) ?? 0;
        return total;
    }

    /**
     * IS-weighted joint posterior over end-of-day survivor sets.
     * Each key is a BigInt bitmask of surviving item IDs (bit i set ⟺ item i survives);
     * by construction every key has popcount = n_end. The map is sparse: only sets
     * actually visited by rollouts appear, so memory is O(rollouts) in the worst case.
     */
    survivorSetPosterior() {
        return { rolloutMap: this.rolloutMap, nWords: this._nWords };
    }

    // ------------------------------------------------------------------
    // MCTS phases
    // ------------------------------------------------------------------

    _iterate() {
        let [node, depth] = this._select(this.root, 0);
        // Only expand if under the node budget (prevents unbounded memory growth)
        if (!node.isTerminal(this.nSells) && depth < this.maxDepth
                && this._nodeCount < this.maxNodes) {
            node = this._expand(node);
            this._nodeCount++;
            depth++;
        }
        const result = this._rollout(node);
        this._backpropagate(node, result);
    }

    /**
     * Hybrid expansion UCB descent.
     *
     * Three phases based on k = node.children.length:
     *
     *   k < eagerCap          → always expand (eager breadth-first phase)
     *   eagerCap <= k < maxCh → opportunistic: expand option competes via UCB
     *                           expandScore = c * sqrt(log(N) / (1 + k))
     *   k >= maxCh            → never expand; pure UCB descent
     *
     * Math.log(parent.visits) is cached once per level.
     */
    _select(node, depth) {
        const c         = this.cExplore;
        const eagerCap  = this.eagerCap;
        const maxCh     = this.maxChildren;
        const maxD      = this.maxDepth;
        const nSells    = this.nSells;

        while (node.sellCount < nSells && depth < maxD) {
            const k = node.children.length;

            // Phase 1: eager — always expand
            if (k < eagerCap) return [node, depth];

            // Phases 2 & 3: need UCB scores for existing children
            const logParentVis = Math.log(node.visits);

            let bestChild = node.children[0];
            let bestScore = bestChild.visits === 0
                ? Infinity
                : c * Math.sqrt(logParentVis / (1 + bestChild.visits));

            for (let i = 1; i < k; i++) {
                const ch = node.children[i];
                const score = ch.visits === 0
                    ? Infinity
                    : c * Math.sqrt(logParentVis / (1 + ch.visits));
                if (score > bestScore) { bestScore = score; bestChild = ch; }
            }

            // Phase 2: opportunistic — expand option competes with best child
            if (k < maxCh) {
                const expandScore = c * Math.sqrt(logParentVis / (1 + k));
                if (expandScore >= bestScore) return [node, depth];
            }

            // Phase 3 (or lost competition): descend
            node = bestChild;
            depth++;
        }
        return [node, depth];
    }

    _expand(node) {
        const numIds = this._numIds;
        let child;
        if (this.rng() > this.qCS) {
            child = new MCTSNode(
                applySell(node.state, this.params, this.rng),
                node.sellCount + 1,
                node.logW + this.logWSell,
                SELL, node, numIds);
        } else {
            child = new MCTSNode(
                applyCompleteShuffle(node.state, this.rng),
                node.sellCount,
                node.logW + this.logWCS,
                COMPLETE_SHUFFLE, node, numIds);
        }
        node.children.push(child);
        return child;
    }

    /**
     * Complete remaining sells nRollouts times using the TRUE distribution P.
     * Returns the reusable Float64Array buffer (valid until next _rollout call).
     * No Set, no flat() — state is scanned directly.
     *
     * Also records each rollout's terminal survivor set into this.rolloutMap with
     * IS weight w = exp(node.logW), so the root carries the joint posterior over
     * end-of-day survivor sets in addition to the per-item marginals.
     */
    _rollout(node) {
        const buf       = this._survivalBuf;
        const keyBuf    = this._keyBuf;
        const nWords    = this._nWords;
        const rmap      = this.rolloutMap;
        const w         = Math.exp(node.logW);
        buf.fill(0);
        const remaining = this.nSells - node.sellCount;
        const params    = this.params;
        const rng       = this.rng;
        const nRollouts = this.nRollouts;

        for (let r = 0; r < nRollouts; r++) {
            let state = node.state;
            for (let s = 0; s < remaining; s++) {
                state = applyTransition(state, params, rng);
            }
            // Direct scan — each item ID appears at most once in state.
            // Simultaneously builds (a) per-item survival tally for marginals
            // and (b) bitmask of surviving IDs for the joint posterior.
            keyBuf.fill(0);
            for (let i = 0; i < state.length; i++) {
                const col = state[i];
                for (let j = 0; j < col.length; j++) {
                    const id = col[j];
                    buf[id]++;
                    keyBuf[id >>> 5] |= 1 << (id & 31);
                }
            }
            // Pack keyBuf (Uint32Array, little-endian word order) into a BigInt key.
            // BigInts are compared by value as Map keys, so identical survivor sets collide.
            let key = 0n;
            for (let i = nWords - 1; i >= 0; i--) {
                key = (key << 32n) | BigInt(keyBuf[i] >>> 0);
            }
            rmap.set(key, (rmap.get(key) ?? 0) + w);
        }

        const invR   = 1.0 / nRollouts;
        const numIds = this._numIds;
        for (let i = 0; i < numIds; i++) buf[i] *= invR;
        return buf;   // caller reads this before the next _rollout
    }

    /**
     * IS-weighted backpropagation.
     * Tight typed-array loop: no property-name lookup, no Set iteration.
     */
    _backpropagate(node, avgResult) {
        const w         = Math.exp(node.logW);
        const effectiveW = this.nRollouts * w;
        const numIds    = this._numIds;
        let n = node;
        while (n !== null) {
            n.visits++;
            n.weightSum += effectiveW;
            const wss = n.weightedSurvivalSum;   // Float64Array
            for (let i = 0; i < numIds; i++) wss[i] += effectiveW * avgResult[i];
            n = n.parent;
        }
    }
}
