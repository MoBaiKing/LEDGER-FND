# Mathematical properties and their boundaries

For available A and uniform pi, maximize <w,u>-tau KL(w||pi) over the simplex. The Lagrange first-order condition gives u_i-tau(log(w_i/pi_i)+1)+lambda=0. Normalization gives w_i proportional to pi_i exp(u_i/tau), hence masked softmax. Strict positivity of tau gives a unique optimum on available entries.

In the strict mask, each evidence state at each layer is a function of only its own initial evidence and fixed role/position. Inductively, Pair(i,j) can depend only on its own initial slot and z_i,z_j. No pair or global feedback edge exists. Thus its Jacobian with respect to any nonendpoint z_k is zero wherever differentiable. This is a locality property from the four input vectors, not a guarantee of raw-modality isolation: E/X are derived multimodal views. Nonconstant endpoint sensitivity must be checked separately.

Let F_u(w)=<w,u>-tau KL(w||pi), w* maximize F_u, and what maximize F_uhat. If ||u-uhat||infinity<=epsilon, then
F_u(w*)-F_u(what)
= [F_u(w*)-F_uhat(w*)] + [F_uhat(w*)-F_uhat(what)] + [F_uhat(what)-F_u(what)] <= 2epsilon.
The middle term is nonpositive; each outside term is bounded by epsilon because w is a probability distribution. This is a regularized utility objective bound under a fixed prior. It is NOT a classification-risk, Macro-F1 or SOTA bound. These are standard properties, not claimed new theorems.

Exact Shapley weights on an n-player available game are |S|!(n-|S|-1)!/n!. Their telescoping permutation interpretation gives sum(phi)=loss(empty)-loss(A). The implementation enumerates the 16 bitmasks and recomputes the smaller-game coefficients for missing views. The game measures reference-model internal deletion, not real-world causal effects or necessarily the final student's deletion effect.
