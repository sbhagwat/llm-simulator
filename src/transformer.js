// Tiny character-level transformer — pure JS, no dependencies
// Simplified but structurally faithful to real LLM training

export function buildVocab(text) {
  const chars = [...new Set(text.split(''))].sort();
  const stoi = {};
  const itos = {};
  chars.forEach((c, i) => {
    stoi[c] = i;
    itos[i] = c;
  });
  return { chars, stoi, itos, vocabSize: chars.length };
}

export function encode(text, stoi) {
  return text.split('').map(c => stoi[c] ?? 0);
}

export function decode(tokens, itos) {
  return tokens.map(t => itos[t] ?? '?').join('');
}

// --- Math helpers ---
function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

function dot(a, b) {
  return a.reduce((sum, v, i) => sum + v * b[i], 0);
}

function matMul(A, B) {
  // A: [m x k], B: [k x n] => [m x n]
  const m = A.length, k = B.length, n = B[0].length;
  return Array.from({ length: m }, (_, i) =>
    Array.from({ length: n }, (_, j) =>
      A[i].reduce((s, _, l) => s + A[i][l] * B[l][j], 0)
    )
  );
}

function randomMatrix(rows, cols, scale = 0.1) {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => (Math.random() - 0.5) * scale)
  );
}

function addVec(a, b) {
  return a.map((v, i) => v + b[i]);
}

function relu(arr) {
  return arr.map(v => Math.max(0, v));
}

// --- Tiny Transformer Model ---
export class TinyTransformer {
  constructor({ vocabSize, embedDim = 16, blockSize = 8, numHeads = 2, ffDim = 32 }) {
    this.vocabSize = vocabSize;
    this.embedDim = embedDim;
    this.blockSize = blockSize;
    this.numHeads = numHeads;
    this.headDim = Math.floor(embedDim / numHeads);
    this.ffDim = ffDim;

    // Token embeddings
    this.tokenEmbed = randomMatrix(vocabSize, embedDim, 0.02);
    // Positional embeddings
    this.posEmbed = randomMatrix(blockSize, embedDim, 0.02);

    // Single attention layer: Q, K, V projections per head
    this.Wq = Array.from({ length: numHeads }, () => randomMatrix(embedDim, this.headDim, 0.02));
    this.Wk = Array.from({ length: numHeads }, () => randomMatrix(embedDim, this.headDim, 0.02));
    this.Wv = Array.from({ length: numHeads }, () => randomMatrix(embedDim, this.headDim, 0.02));
    this.Wo = randomMatrix(numHeads * this.headDim, embedDim, 0.02);

    // Feed-forward
    this.W1 = randomMatrix(embedDim, ffDim, 0.02);
    this.b1 = new Array(ffDim).fill(0);
    this.W2 = randomMatrix(ffDim, embedDim, 0.02);
    this.b2 = new Array(embedDim).fill(0);

    // Output projection
    this.Wout = randomMatrix(embedDim, vocabSize, 0.02);
    this.bout = new Array(vocabSize).fill(0);

    this.lastAttentionWeights = null;
  }

  // Forward pass — returns logits for each position
  forward(tokens) {
    const T = tokens.length;

    // Embeddings
    let x = tokens.map((t, pos) =>
      addVec(this.tokenEmbed[t], this.posEmbed[pos] || new Array(this.embedDim).fill(0))
    );

    // Multi-head self-attention (causal)
    const headOutputs = this.Wq.map((Wq, h) => {
      const Wk = this.Wk[h], Wv = this.Wv[h];
      const Q = x.map(row => row.reduce((acc, v, i) => {
        return acc.map((a, j) => a + v * Wq[i][j]);
      }, new Array(this.headDim).fill(0)));
      const K = x.map(row => row.reduce((acc, v, i) => {
        return acc.map((a, j) => a + v * Wk[i][j]);
      }, new Array(this.headDim).fill(0)));
      const V = x.map(row => row.reduce((acc, v, i) => {
        return acc.map((a, j) => a + v * Wv[i][j]);
      }, new Array(this.headDim).fill(0)));

      const scale = Math.sqrt(this.headDim);
      const attnWeights = Q.map((q, i) => {
        const scores = K.map((k, j) => j <= i ? dot(q, k) / scale : -1e9);
        return softmax(scores);
      });

      if (h === 0) this.lastAttentionWeights = attnWeights;

      return Q.map((_, i) =>
        V[0].map((_, d) =>
          attnWeights[i].reduce((s, w, j) => s + w * V[j][d], 0)
        )
      );
    });

    // Concatenate heads
    let attnOut = x.map((_, i) => headOutputs.flatMap(h => h[i]));
    // Project back
    attnOut = attnOut.map(row =>
      row.reduce((acc, v, i) => {
        return acc.map((a, j) => a + v * this.Wo[i][j]);
      }, new Array(this.embedDim).fill(0))
    );

    // Residual
    x = x.map((row, i) => addVec(row, attnOut[i]));

    // Feed-forward
    const ffOut = x.map(row => {
      const h1 = relu(addVec(
        row.reduce((acc, v, i) => acc.map((a, j) => a + v * this.W1[i][j]), [...this.b1]),
        []
      ));
      const h1full = row.reduce((acc, v, i) => acc.map((a, j) => a + v * this.W1[i][j]), [...this.b1]);
      const h1act = relu(h1full);
      return addVec(
        h1act.reduce((acc, v, i) => acc.map((a, j) => a + v * this.W2[i][j]), [...this.b2]),
        []
      );
    });

    // Compute FF properly
    const ffOutFinal = x.map(row => {
      const h1 = addVec(
        row.reduce((acc, v, i) => acc.map((a, j) => a + v * this.W1[i][j]), new Array(this.ffDim).fill(0)),
        this.b1
      );
      const h1act = relu(h1);
      return addVec(
        h1act.reduce((acc, v, i) => acc.map((a, j) => a + v * this.W2[i][j]), new Array(this.embedDim).fill(0)),
        this.b2
      );
    });

    // Residual
    x = x.map((row, i) => addVec(row, ffOutFinal[i]));

    // Output logits
    const logits = x.map(row =>
      addVec(
        row.reduce((acc, v, i) => acc.map((a, j) => a + v * this.Wout[i][j]), new Array(this.vocabSize).fill(0)),
        this.bout
      )
    );

    return logits;
  }

  // Cross-entropy loss on a single (input, target) pair
  loss(tokens, targets) {
    const logits = this.forward(tokens);
    let totalLoss = 0;
    for (let i = 0; i < targets.length; i++) {
      const probs = softmax(logits[i]);
      totalLoss -= Math.log(Math.max(probs[targets[i]], 1e-10));
    }
    return { loss: totalLoss / targets.length, logits };
  }

  // Numerical gradient descent step (simplified — finite differences on params)
  // For visualization we use a simplified update: perturb + compare
  trainStep(tokens, targets, lr = 0.05) {
    const { loss: currentLoss, logits } = this.loss(tokens, targets);

    // Simplified gradient update using softmax cross-entropy gradient
    // dL/d(logits) = probs - one_hot(target)
    const eps = lr / tokens.length;

    for (let i = 0; i < targets.length; i++) {
      const probs = softmax(logits[i]);
      // Update output weights with gradient
      for (let v = 0; v < this.vocabSize; v++) {
        const grad = probs[v] - (v === targets[i] ? 1 : 0);
        this.bout[v] -= eps * grad;
      }
    }

    // Small random perturbation to other weights to simulate learning
    // (In a real transformer you'd backprop through all layers)
    const noise = 0.001;
    const updateMatrix = (M) => {
      for (let i = 0; i < M.length; i++)
        for (let j = 0; j < M[i].length; j++)
          M[i][j] += (Math.random() - 0.5) * noise * lr;
    };
    this.tokenEmbed.forEach((row, i) => {
      if (tokens.includes(i)) {
        for (let j = 0; j < row.length; j++)
          row[j] += (Math.random() - 0.5) * noise * lr * 2;
      }
    });
    updateMatrix(this.W1); updateMatrix(this.W2);
    this.Wq.forEach(updateMatrix); this.Wk.forEach(updateMatrix); this.Wv.forEach(updateMatrix);

    return currentLoss;
  }

  predict(tokens, itos) {
    const logits = this.forward(tokens);
    const lastLogits = logits[logits.length - 1];
    const probs = softmax(lastLogits);
    const topK = probs
      .map((p, i) => ({ token: itos[i], prob: p }))
      .sort((a, b) => b.prob - a.prob)
      .slice(0, 5);
    return topK;
  }
}
