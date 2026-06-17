# The Hyperbolic Packing Theorem: a research program

## Setup

Let $G = (V, E)$ be a $k$-clique-regular graph --- every vertex is contained
in finitely many $k$-cliques, and the *ball of radius $r$* around any vertex
has $N(r) \sim C \cdot b^r$ vertices for constants $C, b > 1$.

The prototypical example is the $\{3,7\}$ hyperbolic triangle tiling:
triangles ($k=3$), growth rate $b = \varphi^2 \approx 2.618$.

An **equilateral embedding** of $G$ in $\mathbb{R}^d$ is a map
$f: V \to \mathbb{R}^d$ with all edges having equal length
(or, relaxed: minimizing edge-length variance $\sigma^2$).

The **collision depth** $r(G, d)$ is the largest $r$ such that the
equilateral embedding exists with $\sigma^2 < \varepsilon$ for some fixed $\varepsilon$.

## Main Conjecture

**Conjecture.** For any $k$-clique-regular exponential graph $G$ with growth rate $b$,

$$r(G, d) \asymp \frac{\log \Omega_{d-1}}{\log b}$$

where $\Omega_{d-1} = \frac{2\pi^{d/2}}{\Gamma(d/2)}$ is the surface area of $S^{d-1}$.

Since $\log \Omega_{d-1} \sim \frac{d}{2} \log\!\left(\frac{2\pi e}{d}\right)$,
this gives $r(G, d) \sim \frac{d}{2\log b} \cdot \log\!\left(\frac{2\pi e}{d}\right)$,
which is roughly linear in $d$ for moderate $d$.

## The Solid Angle Argument (upper bound)

**Proposition (upper bound).** Suppose the ring-$r$ boundary of $f(G)$, 
projected radially onto $S^{d-1}$ centered at $f(v_0)$, has angular
separation at least $\theta_r$ between any two vertices. Then:

$$r(G, d) \leq \min\!\left\{r : N(r) \cdot \Omega_{\mathrm{cap}}(\theta_r, d) > \Omega_{d-1}\right\}$$

where $\Omega_{\mathrm{cap}}(\theta, d)$ is the solid angle of a spherical cap
of angular radius $\theta$ on $S^{d-1}$.

**Proof sketch.** If $N(r)$ caps of radius $\theta_r$ are packed on $S^{d-1}$
without overlap, their total solid angle cannot exceed $\Omega_{d-1}$. Overlap
implies self-intersection (collision) of the embedding. $\square$

## The Key Lemma

**Lemma (needed).** For an equilateral embedding of a $k$-clique-regular
exponential graph in $\mathbb{R}^d$, the angular separation satisfies

$$\theta_r \asymp C(k) \cdot b^{-r/(d-1)}$$

for a constant $C(k)$ depending only on the local simplex geometry.

**Consequence.** Substituting into the product:

$$N(r) \cdot \Omega_{\mathrm{cap}}(\theta_r, d)
\approx b^r \cdot \left[C(k) \cdot b^{-r/(d-1)}\right]^{d-1}
= C(k)^{d-1}$$

The product is **asymptotically constant** — exponential growth in $N(r)$
exactly cancels the exponential shrinkage in $\theta_r^{d-1}$.

This means:
- If $C(k)^{d-1} < \Omega_{d-1}$: perfect packing is possible, $r(G,d) = \infty$
- If $C(k)^{d-1} \geq \Omega_{d-1}$: finite collision depth

The **critical dimension** $d^*$ where the transition occurs satisfies
$C(k)^{d^*-1} = \Omega_{d^*-1}$.

## Empirical support

| $(k, d)$ | Structure | $N(5) \cdot \Omega_{\mathrm{cap}}(\theta_5, d) / \Omega_{d-1}$ | $r$ |
|----------|-----------|-------------------------------|-----|
| $(3, 3)$ | $\{3,7\}$ in $\mathbb{R}^3$ | $21.9 / 12.6 = 1.74 > 1$ | $5$ (collision) |
| $(3, 4)$ | $\{3,7\}$ in $\mathbb{R}^4$ | $< 1$ (estimated) | $\geq 9$ |
| $(4, 4)$ | $\{3,3,5\}$ in $\mathbb{R}^4$ | $> 1$ at $r=5$ | $4$ |

The $\{3,7\}$ case at $d=3$ shows the packing first violating at $r=5$
(ratio $= 1.74$), consistent with $r(3,3) = 5$.

## What needs to be proved

1. **The Lemma**: $\theta_r \asymp C(k) \cdot b^{-r/(d-1)}$.
   This is the core geometric claim. It requires showing that the ring-$r$
   boundary of an optimally embedded $k$-simplex tiling is
   equidistributed on $S^{d-1}$ at the right scale.
   The connection: equidistribution of the $\{3,7\}$ boundary is related
   to the equidistribution of horocycles in $H^2$ (Ratner's theorem).

2. **The lower bound**: $r(G, d) \geq c \cdot \log \Omega_{d-1} / \log b$.
   This requires constructing explicit embeddings achieving depth $r$.
   The physics relaxer provides computational evidence; an analytic
   construction is open.

3. **The constant $C(k)$**: explicit formula in terms of the dihedral
   angle of the $k$-simplex.

4. **Universality**: the same $r(k, b, d)$ bound holds for any graph
   with local clique size $k$ and growth rate $b$, not just the
   regular hyperbolic tilings. This is the version relevant for
   applications to real networks.

## Connection to representation learning

A concept graph with local clique size $k$ and branching rate $b$
embedded in $\mathbb{R}^d$ can faithfully represent at most $r(k, b, d)$
rings of context around any concept. This gives a principled
**context capacity** bound depending on the graph's local regularity
and the embedding dimension.

