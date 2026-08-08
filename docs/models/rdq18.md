# RDQ18

A reduced-order Ordinary Differential Equation (ODE) model of sarcomere dynamics proposed by
Regazzoni, Dedè, and Quarteroni (2018).

Spatially explicit Markov Chain models of the sarcomere accurately capture length-dependent activation and nearest-neighbor cooperative interactions (such as attached crossbridges increasing the affinity of troponin C to calcium). However, these full models involve an intractable number of degrees of freedom (on the order of $10^{21}$) and require slow Monte Carlo simulations.

The RDQ18 model overcomes this by using a physically motivated assumption of conditional independence to track joint probabilities of triplets of consecutive units. This condenses the system to roughly 2,200 variables, resulting in a system of ODEs that solves 10,000 times faster than the original Monte Carlo method while maintaining high accuracy.

**Reference:** {cite}`regazzoni2018active`
