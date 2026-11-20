// Main-thread helpers copied from the FreshGuard prototype (index.html).
export function popcount32(x) {
    x = x >>> 0;
    x = x - ((x >>> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
    return (((x + (x >>> 4)) & 0x0F0F0F0F) * 0x01010101) >>> 24;
}

export function runMCSimulationAggregate(survivorPosterior, nItems, distroNorm, nMC, swapRatio) {
    // The worker has already weighted-sampled the posterior down to K
    // survivor sets (see sampleSurvivorKeys in worker.js), so this function
    // just iterates {keys, weights, nWords} directly.
    const { keys: sampledKeys, weights: sampledWeights, nWords } = survivorPosterior;
    const K = sampledKeys.length;

    // Build CDF for sampling expiries
    const expiryVals = distroNorm.map(d => d.expiry);
    const cdf = []; let cum = 0;
    for (const d of distroNorm) { cum += d.prob; cdf.push(cum); }
    const nExpiries = expiryVals.length;
    // Index of expiry value 0 (today); -1 if today is not in the distribution
    const todayIdx = expiryVals.indexOf(0);

    function sampleExpiryIdx() {
        const r = Math.random();
        for (let i = 0; i < cdf.length; i++) if (r < cdf[i]) return i;
        return cdf.length - 1;
    }

    // ── Step 1: pre-generate the MC pool as packed bitmasks ──────────────────
    // expMasks[d * nMC * nWords + t * nWords + w] = bit i set iff position i
    // has expiry expiryVals[d] in trial t. One Uint32Array, indexed flat for
    // cache friendliness.
    const expMasks   = new Uint32Array(nExpiries * nMC * nWords);
    const trialStride = nMC * nWords;
    const assignment = new Int32Array(nItems);   // expiry index, not value

    for (let t = 0; t < nMC; t++) {
        // Sample nItems expiry indices
        for (let i = 0; i < nItems; i++) assignment[i] = sampleExpiryIdx();

        // Sort ascending → FIFO (position 0 gets the smallest expiry index)
        assignment.sort();

        // Partial Fisher-Yates: each position i swaps with random j > i
        // with probability swapRatio
        if (swapRatio > 0) {
            for (let i = 0; i < nItems - 1; i++) {
                if (Math.random() < swapRatio) {
                    const j = i + 1 + ((Math.random() * (nItems - 1 - i)) | 0);
                    const tmp = assignment[i]; assignment[i] = assignment[j]; assignment[j] = tmp;
                }
            }
        }

        // Pack into per-expiry bitmasks
        const tOff = t * nWords;
        for (let i = 0; i < nItems; i++) {
            const d = assignment[i];
            expMasks[d * trialStride + tOff + (i >>> 5)] |= 1 << (i & 31);
        }
    }

    // ── Step 2: evaluate over (sampled S, w_S) × MC trials with popcount ──────
    // sampledKeys carries K survivor sets drawn from the posterior; weights
    // are 1 for multinomial samples (>K case) or the original IS weights when
    // we kept all entries (<=K case). Either way the final normalization just
    // divides by Σ (w_S × nMC).
    const pmfWeight = new Map();
    const rSum   = new Float64Array(nExpiries);
    const rSumSq = new Float64Array(nExpiries);
    let W = 0;   // Σ w_S × nMC (denominator)

    const Sbuf = new Uint32Array(nWords);
    const ALL_ONES_32 = 0xFFFFFFFFn;

    for (let s = 0; s < K; s++) {
        const w_S = sampledWeights[s];

        // Unpack BigInt → Uint32Array (little-endian word order)
        let k = sampledKeys[s];
        for (let i = 0; i < nWords; i++) {
            Sbuf[i] = Number(k & ALL_ONES_32);
            k >>= 32n;
        }

        W += w_S * nMC;

        for (let t = 0; t < nMC; t++) {
            const tOff = t * nWords;

            // PMF over today-expiring survivors
            if (todayIdx >= 0) {
                let n_today = 0;
                const dOff = todayIdx * trialStride + tOff;
                for (let w = 0; w < nWords; w++) {
                    n_today += popcount32(Sbuf[w] & expMasks[dOff + w]);
                }
                pmfWeight.set(n_today, (pmfWeight.get(n_today) ?? 0) + w_S);
            } else {
                // No expiry-today bucket → all survivors fall in the "0 today" bin
                pmfWeight.set(0, (pmfWeight.get(0) ?? 0) + w_S);
            }

            // Per-expiry remain stats: count survivors with expiry d, accumulate
            // first and second moments (weighted by w_S).
            for (let d = 0; d < nExpiries; d++) {
                let n_d = 0;
                const dOff = d * trialStride + tOff;
                for (let w = 0; w < nWords; w++) {
                    n_d += popcount32(Sbuf[w] & expMasks[dOff + w]);
                }
                rSum[d]   += w_S * n_d;
                rSumSq[d] += w_S * n_d * n_d;
            }
        }
    }

    // Normalize
    const pmf = new Map();
    if (W > 0) {
        for (const [k, v] of pmfWeight) pmf.set(k, v / W);
    }

    const remainStats = {};
    for (let d = 0; d < nExpiries; d++) {
        const mean   = W > 0 ? rSum[d]   / W : 0;
        const meanSq = W > 0 ? rSumSq[d] / W : 0;
        remainStats[expiryVals[d]] = {
            mean,
            std: Math.sqrt(Math.max(0, meanSq - mean * mean)),
        };
    }
    return { pmf, remainStats };
}

export function normalizeDistroFromJSON(distribution) {
    const totalW = distribution.reduce((s, e) => s + e.weight, 0);
    if (totalW === 0) return [];
    return distribution.map(e => ({ expiry: e.expiry, prob: e.weight / totalW }));
}

export function sampleLayout(nItems, nCols, strat, colSizes) {
    const cols = Array.from({ length: nCols }, () => []);
    if (strat === 'roundrobin') {
        // Round-robin: items dealt one-at-a-time across columns
        for (let i = 0; i < nItems; i++) cols[i % nCols].push(i);
    } else if (strat === 'colwise') {
        // Column-wise: fill each column to capacity before moving to next
        const caps = colSizes ?? (() => {
            const base = Math.floor(nItems / nCols), extra = nItems % nCols;
            return Array.from({ length: nCols }, (_, c) => base + (c < extra ? 1 : 0));
        })();
        let id = 0;
        for (let c = 0; c < nCols; c++) {
            for (let k = 0; k < caps[c]; k++) cols[c].push(id++);
        }
    } else {
        // Uniform random
        for (let i = 0; i < nItems; i++) cols[(Math.random() * nCols) | 0].push(i);
    }
    return cols.filter(c => c.length > 0);
}

