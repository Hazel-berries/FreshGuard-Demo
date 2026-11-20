import { InventoryMCTS } from './mcts.js';

// Cap on how many survivor sets we ship across the worker boundary. Sampled
// here so the postMessage payload (and the main thread's peak memory) stays
// bounded regardless of how diverse the posterior is. Each sampled key is
// equally-weighted at the call site (1/K).
const SURVIVOR_SAMPLE_CAP = 10000;

// Multinomial sampling with replacement from Map<key, weight>, via CDF +
// binary search. Returns parallel arrays { keys, weights }: when N <= K we
// keep all entries with their original IS weights; when N > K we draw K
// samples weighted by w (equally-weighted thereafter).
function sampleSurvivorKeys(rolloutMap, K) {
    const N = rolloutMap.size;
    if (N === 0) return { keys: [], weights: new Float64Array(0) };
    if (N <= K) {
        const keys = new Array(N);
        const weights = new Float64Array(N);
        let i = 0;
        for (const [k, w] of rolloutMap) { keys[i] = k; weights[i] = w; i++; }
        return { keys, weights };
    }
    const keys = new Array(N);
    const cdf  = new Float64Array(N);
    let cum = 0, i = 0;
    for (const [k, w] of rolloutMap) { cum += w; cdf[i] = cum; keys[i] = k; i++; }
    const sampledKeys    = new Array(K);
    const sampledWeights = new Float64Array(K);
    for (let s = 0; s < K; s++) {
        const r = Math.random() * cum;
        let lo = 0, hi = N - 1;
        while (lo < hi) {
            const mid = (lo + hi) >>> 1;
            if (r < cdf[mid]) hi = mid;
            else lo = mid + 1;
        }
        sampledKeys[s] = keys[lo];
        sampledWeights[s] = 1;
    }
    return { keys: sampledKeys, weights: sampledWeights };
}

function sampleLayout(nItems, nCols, strat) {
    const cols = Array.from({ length: nCols }, () => []);
    if (strat === 'roundrobin') {
        for (let i = 0; i < nItems; i++) cols[i % nCols].push(i);
    } else if (strat === 'colwise') {
        const base = Math.floor(nItems / nCols);
        const extra = nItems % nCols;
        let id = 0;
        for (let c = 0; c < nCols; c++) {
            const cap = base + (c < extra ? 1 : 0);
            for (let k = 0; k < cap; k++) cols[c].push(id++);
        }
    } else {
        for (let i = 0; i < nItems; i++) cols[(Math.random() * nCols) | 0].push(i);
    }
    return cols.filter(c => c.length > 0);
}

self.onmessage = ({ data }) => {
    // Multi-layout mode (Tab 5): run nLayouts MCTS runs and average probs
    if (data.nLayouts != null) {
        const { nItems, nCols, strat, nLayouts, params, mctsOptions, nIter } = data;
        const K = data.survivorSampleCap ?? SURVIVOR_SAMPLE_CAP;
        const totalIter = nLayouts * nIter;
        const probsList = [];
        let totalVisits = 0;

        // Survivor-set posteriors are merged INCREMENTALLY: after each layout's
        // MCTS finishes we fold its rolloutMap into mergedMap and let the layout
        // MCTS go out of scope, so peak memory is one MCTS at a time (not 10).
        // Each layout's map is normalized to sum=1 then averaged with weight
        // 1/nLayouts, matching how marginal probs are averaged below. Item IDs
        // are exchangeable across layouts (just position indices), so identical
        // bitmask keys aggregate correctly across layouts.
        const mergedMap = new Map();
        let mergedNWords = 0;

        for (let l = 0; l < nLayouts; l++) {
            const initialState = sampleLayout(nItems, nCols, strat);
            let mcts = new InventoryMCTS(initialState, params, mctsOptions);
            const baseIter = l * nIter;
            mcts.run(nIter, {
                onProgress: (done, _total) =>
                    self.postMessage({ type: 'progress', done: baseIter + done, total: totalIter }),
            });
            probsList.push(Object.fromEntries(mcts.survivalProbabilities()));
            totalVisits += mcts.root.visits;

            // Fold this layout's posterior into mergedMap, then drop it.
            const { rolloutMap, nWords } = mcts.survivorSetPosterior();
            if (nWords > mergedNWords) mergedNWords = nWords;
            let W_l = 0;
            for (const v of rolloutMap.values()) W_l += v;
            if (W_l > 0) {
                const factor = 1 / (W_l * nLayouts);
                for (const [key, w] of rolloutMap) {
                    mergedMap.set(key, (mergedMap.get(key) ?? 0) + w * factor);
                }
            }
            // Release this layout's MCTS tree, rolloutMap, and per-node arrays.
            mcts = null;
        }

        // Average probs across all layouts
        const allKeys = new Set(probsList.flatMap(p => Object.keys(p)));
        const avgProbs = {};
        for (const k of allKeys) {
            avgProbs[k] = probsList.reduce((s, p) => s + (p[k] ?? 0), 0) / nLayouts;
        }

        // Sample mergedMap down to K before crossing the worker boundary, so
        // the postMessage payload (and main-thread peak memory) is bounded.
        const sampled = sampleSurvivorKeys(mergedMap, K);
        self.postMessage({
            type: 'done',
            probs: avgProbs,
            survivorPosterior: { ...sampled, nWords: mergedNWords },
            visits: totalVisits,
        });
        return;
    }

    // Single-layout mode (Tab 2 / Tab 3)
    const { initialState, params, mctsOptions, nIter } = data;
    const K = data.survivorSampleCap ?? SURVIVOR_SAMPLE_CAP;
    const mcts = new InventoryMCTS(initialState, params, mctsOptions);
    mcts.run(nIter, {
        onProgress: (done, total) => self.postMessage({ type: 'progress', done, total }),
    });
    const { rolloutMap, nWords } = mcts.survivorSetPosterior();
    const sampled = sampleSurvivorKeys(rolloutMap, K);
    self.postMessage({
        type: 'done',
        probs: Object.fromEntries(mcts.survivalProbabilities()),
        survivorPosterior: { ...sampled, nWords },
        visits: mcts.root.visits,
    });
};
