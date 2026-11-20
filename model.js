/**
 * Generative model for perishable inventory on a multi-column shelf.
 *
 * State:  array of columns, each column an array of item IDs ordered front-to-back.
 * Events (superposition of two Poisson processes, thinned each step):
 *   - Sell + limited shuffle  (probability p_sell)
 *   - Complete shuffle        (probability 1 - p_sell)
 */

/** Mulberry32 seeded PRNG — returns floats in [0, 1). */
export function createRNG(seed = Date.now()) {
    let s = (seed >>> 0) || 1;
    return () => {
        s += 0x6D2B79F5;
        let t = s ^ (s >>> 15);
        t = Math.imul(t, 1 | s);
        t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

export function totalItems(state) {
    let n = 0;
    for (let i = 0; i < state.length; i++) n += state[i].length;
    return n;
}

/**
 * Sample position k in [0, colLen-1] with weights proportional to (1-q)^k.
 * No temporary array allocation: two sequential passes over [0, colLen).
 */
function sampleGeomPosition(colLen, q, rng) {
    const onemq = 1.0 - q;
    let w = 1.0, total = 0.0;
    for (let k = 0; k < colLen; k++) { total += w; w *= onemq; }
    let r = rng() * total;
    w = 1.0;
    for (let k = 0; k < colLen; k++) {
        r -= w;
        if (r <= 0.0) return k;
        w *= onemq;
    }
    return colLen - 1;
}

/** In-place Fisher-Yates shuffle. */
function fisherYatesShuffle(arr, rng) {
    for (let i = arr.length - 1; i > 0; i--) {
        const j = (rng() * (i + 1)) | 0;
        const tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

/**
 * Sell + limited shuffle event. Returns a new state (does not mutate input).
 *
 * 1. Choose a non-empty column uniformly.
 * 2. Choose position k ~ Geometric(q) truncated to column length.
 * 3. Remove item at k (sold).
 * 4. H = items at positions [0, k) that were displaced.
 * 5. With prob p_s: shuffle H randomly.
 * 6. Each item in H independently: with prob p_r, unshift into a randomly chosen
 *    adjacent column; otherwise push into stay[], which becomes the new front of col c.
 * 7. Tail (positions k+1..) is appended behind stay[].
 */
export function applySell(state, params, rng) {
    const nCols = state.length;
    // Deep copy — columns will be mutated
    const cols = new Array(nCols);
    for (let i = 0; i < nCols; i++) cols[i] = state[i].slice();

    // Count and pick a non-empty column uniformly without building a helper array
    let nonEmptyCount = 0;
    for (let i = 0; i < nCols; i++) if (cols[i].length > 0) nonEmptyCount++;
    if (nonEmptyCount === 0) return state;

    let pick = ((rng() * nonEmptyCount) | 0) + 1;
    let c = 0;
    for (let i = 0; i < nCols; i++) {
        if (cols[i].length > 0 && --pick === 0) { c = i; break; }
    }

    const col = cols[c];
    const k = sampleGeomPosition(col.length, params.q, rng);

    // H = displaced items; tail = items behind the sold position
    const H    = col.slice(0, k);
    const tail = col.slice(k + 1);

    if (rng() < params.p_s) fisherYatesShuffle(H, rng);

    const hasLeft  = c > 0;
    const hasRight = c < nCols - 1;
    const canRedist = hasLeft || hasRight;
    const p_r = params.p_r;

    const stay = [];
    for (let i = 0; i < H.length; i++) {
        const item = H[i];
        if (canRedist && rng() < p_r) {
            // Pick adjacent column without allocating a neighbors array
            const target = (hasLeft && hasRight)
                ? (rng() < 0.5 ? c - 1 : c + 1)
                : (hasLeft ? c - 1 : c + 1);
            cols[target].unshift(item);   // insert at front (cols[target] already copied)
        } else {
            stay.push(item);
        }
    }
    // Append tail to stay without spread
    for (let i = 0; i < tail.length; i++) stay.push(tail[i]);
    cols[c] = stay;
    return cols;
}

/**
 * Complete shuffle: pool all items, assign each uniformly to a random column.
 * Per-column counts may change; within-column order is random.
 */
export function applyCompleteShuffle(state, rng) {
    const nCols = state.length;
    let totalCount = 0;
    for (let i = 0; i < nCols; i++) totalCount += state[i].length;
    if (totalCount === 0) return state;

    // Collect all IDs into a pre-sized array (avoids flat())
    const allIds = new Array(totalCount);
    let idx = 0;
    for (let i = 0; i < nCols; i++) {
        const col = state[i];
        for (let j = 0; j < col.length; j++) allIds[idx++] = col[j];
    }

    fisherYatesShuffle(allIds, rng);

    const newCols = new Array(nCols);
    for (let i = 0; i < nCols; i++) newCols[i] = [];
    for (let i = 0; i < totalCount; i++) {
        newCols[(rng() * nCols) | 0].push(allIds[i]);
    }
    return newCols;
}

/**
 * Advance by exactly one sell event, absorbing all preceding complete shuffles.
 * Between consecutive sells the number of CS is Geometric(p_sell)-1.
 */
export function applyTransition(state, params, rng) {
    while (rng() > params.p_sell) {
        state = applyCompleteShuffle(state, rng);
    }
    return applySell(state, params, rng);
}
