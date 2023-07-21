# Quantitative Investment analysis


# Lec1 Mean-variance portfolio choice
## Basic concepts of portfolios
- Gross return 
  $$
    R = Gross \enspace return = \frac{amount \enspace recevied}{amount \enspace invested}
  $$
- Rate of return
  $$
    r = Rate \enspace of \enspace return = \frac{amount \enspace recevied - amount \enspace invested}{amount \enspace invested} = R-1
  $$
- Amount recevied = net dividend and repurchase received during the investment + terminal asset payoffs
- Short sales \
    You receive $X_0$ initially and pay $X_1$ + D later, so your initial cost is $-X_0$ and the final payoff is $-X_1$ - D, and hence your gross return is given by,
    $$
      R = \frac{amount \enspace received}{amount \enspace invested} = \frac{-X_1 - D}{-X_0} = \frac{X_1 + D}{X_0} = r+1
    $$
- Random payoffs(r) \
  In real word, both intermediate cash flows and terminal payoffs are uncertain.

    > &ensp; r: discrete random variable \
     &ensp; n: corresponding probabilities ${p_1, ..., p_n}$, where $\sum _i p_i = 1$ and 0 $\leqslant p_i \leqslant$ 1

    mean = $\overline{r}$ = $\sum\limits_{n=1}^{n} r_i p_i$ \
    variance = $\sigma^2_r$ = $\sum\limits_{n=1}^{n} (r_i - \overline{r})^2 p_i$ \
    standard deviation = $\sigma_r$ = $\sqrt{\sum\limits_{i=1}^{n} (r_i - \overline{r})^2 p_i}$

- Continuous payoffs
  
    > return can be any value in ($-\infty, \infty$) (or (0, $\infty$))

    Cumulative Distribution Function (CDF) of r:
    $$
    F(x) = Pr(r\leqslant x)
    $$

  Probability density function (pdf) of r:
  $$
  f(x) = \lim_{\Delta \to 0} \frac{Prob(x < r \leqslant x+\Delta)}{\Delta} = \lim_{\Delta \to 0} \frac{F(x+\Delta)-F(x)}{\Delta} = \frac{\partial F(t)}{\partial t}|_{t=x}
  $$

  Expectation of r:
  $$
  E[r] = \int^x_{-\infty} t f(t) dt
  $$

  Variance of r:
  $$
  Var[r] = E[(r-E[r])^2] = \int_{-\infty}^{\infty} (t-E[r])^2 f(t)dt
  $$

- Covarience \
  For two discrete random variable r and m, their mutual linear dependence can be characterized by their covariance.

  $$
  Cov(r, m) = E[(r-E[r])(m-E[m])] = E[rm] - E[r]E[m] \\\
  Symmetry: Cov(r, m) = Cov(m, r)
  $$

  If r and u are both discrete rv,
  $$
  Cov(r,m)=\sum\limits_{i=1}^{n} \sum\limits_{j=1}^{n} {(r_i - \overline{r})(u_j - \overline{u})p_{ij}}
  $$
  where,
  $$
  p_{ij} = Prob(r=r_i \\& u=u_j)
  $$

  If r and u are both Continuous rv, we have
  $$
  Cov(r,u) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} (x-E[r])(y-E[m])f(x,y)dxdy
  $$

  where f is the joint probility density function of (r,m).

- Correlation
  $Cov(r_1, r_2) = 0$, they are said to be uncorrelated and independent.
  $$
  E[r_1, r_2] = \overline{r_1} \enspace \overline{r_2} \longrightarrow Cov(r_1, r_2)
  $$
  
  The correlation coefficient of a pair of random variavles:
  $$
  \rho_{r_1, r_2} = \frac{Cov(r_1, r_2)}{\sigma_{r_1} \sigma_{r_2}}
  $$
  $$
  |\rho_{r_1, r_2}| \leqslant 1 \longrightarrow |Cov(r_1, r_2)| \leqslant \sigma_{r_1} \sigma_{r_2}
  $$

- Portfolio
  Weight:

  > &ensp; Initial wealth $W_0$ \
    &ensp; $W_{0i}$ in each asset \
    &ensp; total $\sum_i W_{0i} = W_0$, $W_{0i}$ can be negative

  Define:
  $$
  \sum\limits_{i} w_i = 1
  $$
  In matrix form, $e^{'}w =1 $, where e i a n $\times$ 1 column vector of 1.

  $e^{'}w =0 $: 0成本组合，long-short组合

  Portfolio return:

  > &ensp; $R_i$ : the gross return of asset i \
    &ensp; $R_i W_{0i} = R_{i \omega_i}W_0$

  Gross return of the portfolio is:
  $$
  R_p = \frac{amount \enspace received}{amount \enspace invested} = \frac{\sum_i R_{i \omega_i}W_0}{W_0} = \sum\limits_{i} \omega_i R_i = \omega^{'}R
  $$
  Rate of return r is:
  $$
  r_p = R_p -1 = \sum\limits_{i} \omega_i R_i -1 = \sum\limits_{i} \omega_i (R_i -1) = \sum\limits_{i} \omega_i r_i = \omega^{'}r
  $$

- Excess Return and Long-short Portfolio
  
  > &ensp; $R_p = \omega_p^{'} R$ \
    &ensp; $R_q = \omega_q^{'} = R$ \
    &ensp; $R_f$ : Risk-free return

  Portfolio p return in excess of risk-free rate: $R_p - R_f = \omega_p^{'}R - \omega_p^{'}(R_f e)$ \
  Portfolio p return excess in excess of q: $R_p - R_q = \omega_p^{'} R - \omega_q^{'}R$ \
  $W_p - W_q$ =0 $\longrightarrow$ zero cost stratry

  Denote:
  $$
  \overline{r} =  \left (\begin{array}{cc} \overline{r_1} \\ \overline{r_2} \\ . \\ . \\ \overline{r_n} \end{array} \right)
  $$
  The expected rate of return of the portfolio:
  $$
  E[r_p] = E[\sum\limits_{i} w_i r_i] = \sum\limits_{i} w_i E[r_i] = \sum\limits_{i} w_i \overline{r_i} = \omega^{'}\overline{r}
  $$
  The variance of the portfolio $\sigma_p^2$:
  $$
  \begin{aligned}
  \sigma_p^2 &= E[(r_p - \overline{r_p})^2]\\\
             &= E[(\sum\limits_{i} \omega_i r_i - \sum\limits_{i} \omega_i \overline{r_i})^2]\\\
             &= E[(\omega^{'}r - \omega^{'}\overline{r})(\omega^{'}r - \omega^{'} \overline{r})]\\\
             &= E[\omega^{'}(r-\overline{r})(r-\overline{r})^{'} \omega]\\\
             &= \omega^{'}E[(r-\overline{r})(r-\overline{r})^{'}]\omega \enspace(\omega, a \enspace constant \enspace vevtor, \enspace canbe\enspace  taken\enspace  out) \\\
             &= \omega^{'} V \omega, \enspace where \enspace V_{i,j} = \sigma_{ij} 
  \end{aligned}
  $$

- OLS Estimator for Linear Regression
  
  > &ensp; Formula: $Y=X \beta + \epsilon$ \
    &ensp; OLS estimator: $\hat{\beta_{OLS}} = \mathop{\arg\min}_{\beta} (Y-X \beta)^{'} (Y-X \beta)$

  $$
  Y = \left(
  \begin{matrix}
  y_1 \\\
  y_2 \\\
  . \\\
  . \\\
  y_n \\\
  \end{matrix}
  \right) +
  \left(
  \begin{matrix}
  1 & E_{du_1} & E_{xp_1} & I_{Q_1} \\\
  1 & E_{du_2} & E_{xp_2} & I_{Q_2} \\\
  . & . & . & . \\\
  . & . & . & . \\\
  1 & E_{du_n} & E_{xp_n} & I_{Q_n} \\\
  \end{matrix}
   \right) +
   \left(
   \begin{matrix}
   \epsilon_1 \\\
   \epsilon_2 \\\
   . \\\
   . \\\
   \epsilon_n \\\
   \end{matrix}
   \right)
  $$

  $$
  \min \limits_{\beta} \epsilon^T \epsilon = \sum \epsilon_i^2 = f(\beta) \\\
  \downarrow \\\
  \frac{\partial f(\beta)}{\partial \beta} = 0
  $$

  Because:

  $$
  \begin{aligned}
   (Y-X \beta)^T (Y-X \beta) &= (Y^T - \beta^T X^T)(Y - X \beta)\\\
                             &= Y^T - 2Y^T X\beta + \beta^T X^T X \beta
  \end{aligned}
  $$
  So:
  $$
  \frac{\partial f(\beta)}{\partial \beta} = 0 \\\
  \downarrow \\\
  0 - 2X^TY + \beta X^T X \hat{\beta} = 0 \Longrightarrow \hat{\beta} = (X^T X)^{-1} X^T Y 
  $$

- Diversification
  
  > &ensp; If the rate of returns on individual assets are correlated
  
  $$
  Var(\tilde{r_p})=\sigma_p=\sum\limits_{n=1}^{\infty} a_n z^n
  $$

  $$
  \begin{align}
  Var(\tilde{r_p}) = \sigma_p^2 &= \sum\limits_{i} \sum\limits_{j} \frac{1}{n^2} \sigma_{ij} \\\
  &= \frac{1}{n^2} \\{ \sum\limits_{i=1}^{n} \sigma_i^2 + \sum\limits_{i\neg j} \sigma_{ij} \\} \\\
  &= \frac{1}{n^2} \\{ n\sigma_{Avg}^2 + (n^2 - n)\sigma_{ij}^{Avg}  \\} \\\
  &= \frac{1}{n} \underbrace{\\{ \sigma_{Avg}^2 - \sigma_{ij}^{Avg} \\} }\_{\text{if bounded}} + \sigma_{ij}^{Avg} \longrightarrow \sigma_{ij}^{Avg}
  \end{align}
  $$

  > &ensp; The variance of a portfolio with a large number of individual assets is mainly determined by the covariance of the individual assets.

  A rational risk-averse investor likes return and dislike risk. She should choose a portfolio of assets that efficiently balance expected returns and risks.

## Minimal-variance analysis
### Two risky assets
  
  > &ensp; Let $\overline{R}_A < \overline{R}_B$ and $\sigma_A^2 < \sigma_B^2$ \
   &ensp; Form a portfolio p with a proportion $\omega \enspace (1-\omega)$ invested in asset A(B)
  
  The expected return and standard deviation of the portfolio:
  $$
  \overline{R}_p = \omega \overline{R}_A + (1-\omega)\overline{R}_B = \overline{R}_B - \omega (\overline{R}_B - \overline{R}_A)\\\
  \sigma_p = [\omega^2 \sigma_A^2 + 2\rho \omega (1-\omega)\sigma_A \sigma_B + (1-\omega)^2 \sigma_B^2]^\frac{1}{2}
  $$
  consider an extreme case $\rho=1, \Longrightarrow =|\omega \sigma_A +(1-\omega_B)|$ \
  We can solve for:
  $$
  \omega = \frac{\sigma_B \pm \sigma_p}{\sigma_B - \sigma_A} \\\
  \begin{aligned}
   \overline{R}_p = \overline{R}_B + \omega &= \frac{\sigma_B \pm \sigma_p}{\sigma_B - \sigma_A} (\overline{R}_B-\overline{R}_A) \\\
   &= \frac{\sigma_B \overline{R}_A-\sigma_A \overline{R}_B}{\sigma_B - \sigma_A} \pm \frac{\overline{R}_B - \overline{R}_A}{\sigma_B - \sigma_A}\sigma_p
  \end{aligned}
  $$

  ![Alt](/images/Efficient_Frontier.jpg  "Efficient_Frontier")

### Multiple risky assets
  
  > &ensp; V is invertible (no redundant assets)
  
  The Markowitz's mean-variance portfolio proble:
  $$
  \min_{\omega} \frac{1}{2} \omega^{'}V \omega \\\
  st. e^{'} \omega = 1\\\
  \omega^{'} \overline{R} = \overline{R}_p
  $$
  
  Form the Lagrangian: \
  $$
  \min_{\omega, \lambda, \gamma} L = \underbrace{\frac{1}{2} \omega^{'}V \omega }_{\frac{1}{2}存在与否都可以} + \lambda (\overline{R}_p - \omega^{'} \overline{R}) + \gamma(1-\omega^{'}e)
  $$

  FOC w.r.t. $\omega$ and two Lagrange multipliers, \
  $\omega - \lambda \overline{R} - \gamma e = 0, \Longrightarrow $ 
  
  $$
  \omega^{*} = \underbrace{\lambda}_{权重1} V^{-1} \overline{R} + \underbrace{\gamma}\_{权重2} V^{-1} e
  $$
  
  > &ensp; Two fund separation theorem: the MVF is a linear combination of two canonic portfolios: $V^{-1}\overline{R}$ and $V^{-1}e$. An inverstor's risk preference (related to $\lambda$ and $\gamma$) determines the weights on these two funds.

  Solve the Largange multipliers(带入$\omega^{*}$): 
  
  $$
  \begin{equation}
  \begin{aligned}
  \overline{R}_p &= \overline{R^{'}} \omega^{*} = \lambda \overline{R^{'}}V^{-1} \overline{R} + \gamma \overline{R^{'}} V^{-1}e \\\
  1 &= e^{'} \omega = \lambda e^{'}V^{-1} \overline{R} + \gamma e^{'}V^{-1} e \tag{9}
  \end{aligned}
  \end{equation}
  $$

### Portfolio math
  Write the preceding linear equations (9) in a matrix form:

  $$
  \left(
  \begin{matrix}
  \overline{R^{'}}V^{-1} \overline{R} & \overline{R^{'}}V^{-1}e \\\
  \overline{R^{'}}V^{-1}e & e^{-1}V^{-1}e 
  \end{matrix}
  \right)
  \left(
  \begin{matrix}
  \lambda \\\
  \gamma
  \end{matrix}
  \right) = 
  \left(
  \begin{matrix}
  \overline{R}_p \\\
  1
  \end{matrix}
  \right)
  $$
  Solve the two Lagrange multipliers:

  $$
  \lambda = \frac{\delta \overline{R}_p - \alpha}{\Delta} \enspace and \enspace \gamma = \frac{\xi - \alpha \overline{R}_p}{\Delta} \tag{10}
  $$
  Where $\alpha, \delta, \xi$ and $\Delta$ are functions of V and $\overline{R}_p$:
  $$
  \begin{equation}
  \begin{aligned}
  \delta \equiv e^{'} V ^{-1} e > 0, \\\
  \alpha \equiv \overline{R}^{'} V^{-1}e, \\\
  \xi = \overline{R}^{'}V^{-1}\overline{R} > 0, \\\
  \Delta = \delta \xi - \alpha^2 \tag{11}
  \end{aligned}
  \end{equation}
  $$

  The optimal portfolio weights $\omega^{*}$ :

  $$
  \begin{aligned}
  \omega^{*} &= \frac{\delta \overline{R}_p - \alpha}{\Delta} V^{-1} \overline{R} + \frac{\xi - \alpha \overline{R}_p}{\Delta} V^{-1}e \\\
  &= \alpha + b \overline{R}_p
  \end{aligned}
  $$

  > &ensp; 只有在有效前沿上的投资组合的权重才可以用上述表达式
  
  Where

  $$
  \alpha \equiv \frac{\xi V^{-1}e - \alpha V^{-1} \overline{R}}{\Delta} \enspace and \enspace b \equiv \frac{\delta V^{-1}\overline{R} - \alpha V^{-1}e}{\Delta}
  $$

  > &ensp; $\alpha$ and b are fixted vectors, given the investment opportunity set($\overline{R}, V$)   

  The variance of frontier portfolio with the expected return $\overline{R}_p$:

  $$
  \sigma_p^2 = \omega^{'\*}V \omega^{\*} = \omega^{'\*}(\lambda \overline{R}+\gamma e)=\lambda \overline{R}_p + \gamma
  $$

  Substituting the expression of $\lambda$ and $\gamma$: \
  We have 

  $$
  \sigma_p^2 = \frac{1}{\delta}+ \frac{\delta(\overline{R}_p -\frac{\alpha}{\delta})^2}{\Delta} \tag{13}
  $$

  The global minimal variance portfolio (GMV): $\sigma_{mv}^2=\frac{1}{\delta}$ with $\overline{R}_{mv} = \frac{\alpha}{\delta}$.

  $$
  \omega_mv = \alpha + b \overline{R}_{mv} = \frac{V^{-1}e}{\delta} = \frac{V^{-1}e}{e^{'}V^{-1}e}
  $$

  $\frac{\partial \sigma_p^2}{\partial \overline{R}\_p}|\_{\overline{R}_{mv}}=0$ (cannot reduce variance by giving up higher expected returns!) 

  Find the weight on the GMV portfolio in another way:\
  $$
  \min_{\omega} \frac{1}{2} \omega^{'}V\omega \\\
  \enspace \\\
  st. \enspace  e^{'} \omega = 1
  $$

  FOC w.r.t $\omega$ and the Largrange multiplier, we have

  $$
  \omega = \gamma V^{-1} e = \frac{V^{-1}e}{e^{'}V^{-1}e}
  $$

### MVF geometry
  A parabola in ($\overline{R}_p, \sigma_p^2$)

  $$
  \sigma_p^2 = \frac{1}{\delta}+ \frac{\delta(\overline{R}_p -\frac{\alpha}{\delta})^2}{\Delta}
  $$
  ![Alt](/images/MVF.jpg "MVF")

  > &ensp; The global minimum ($\frac{\alpha}{\delta}, \frac{1}{\delta}$) in ($\overline{R}_p, \sigma_p^2$)

  $$
  \frac{\sigma_p^2}{\frac{1}{\delta}}-\frac{\overline{R}_p-\frac{\alpha}{\delta}}{\frac{\Delta}{\delta^2}}=1, \sigma_p > 0
  $$
  ![Alt](\images/asymoptote.jpg "Asymoptote")

  The asymptote: $\lim_{\sigma_p \rightarrow \infty}\frac{d  \overline{R}_p}{d \sigma_p}$

  > &ensp; Definition: The frontier portfolios with expected return higher(lower) than $\overline{R}_{mv}=\frac{\alpha}{\delta} $ are called efficient(inefficient) frontier portfolios.

### Two Fund Theorem
