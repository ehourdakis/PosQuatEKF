## PosQuatEKF
Implementation of an EKF for a position-quaternion state space system.  The project can be compiled by issuing the following commands:

```bash
cmake . -DUSE_GNUPLOT=ON -B build/
cmake --build build/ --parallel -j${nproc}
```

A test can be run using:
```bash
./build/tests/poser/example_poser
```

gnuplot can be installed using:
```bash
sudo apt-get install gnuplot libgnuplot-iostream-dev
```

If gnuplot is enabled, pqekf will generate graphs for debugging:

![Trajectory with covariance](/data/images/trajectory.png "Trajectory with covariance").

pqekf is robust, i.e. it can reject outliers in extreme cases:

![Plots](/data/images/outliers.png "Outlier plot").
![Plots](/data/images/plots_pos.png "Position plot").
![Plots](/data/images/plots_quat.png "Quaternion plot").

## Mathematical Foundations

#### Quaternion state transition
The quaternion is predicted by integrating the angular velocity $\omega$:

```math
\hat{\mathbf{q}}_t = \mathbf{q}_{t-1} + \int_{t-1}^t\boldsymbol\omega\, dt
```

Closed-form solutions of this integration are not available, so an approximation method is required.

#### Derivative approximation
Given an angular velocity vector 

```math
\boldsymbol{\omega} = \begin{bmatrix} \omega_{x} \\ \omega_{y} \\ \omega_{z} \end{bmatrix}
```

and a quaternion 

```math
\boldsymbol{q} = \begin{bmatrix}q_{x} \ q_{y} \ q_{z} \ q_{w}\end{bmatrix}
```

representing the orientation of the system, we want to approximate the derivative. 
The $n^{th}$ order Taylor series expansion of $q(t_n + \Delta t)$ arround time $t=t_n$ is:

```math
\mathbf{q}_t = \mathbf{q}_{t-1} + \dot{\mathbf{q}}_{t-1}\Delta t +
\frac{1}{2!}\ddot{\mathbf{q}}_{t-1}\Delta t^2 +
\frac{1}{3!}\ddot{\mathbf{q}}_{t-1}\Delta t^3 + \cdots

```

We perturb the system with a small angle $\Delta q_L$. By applying the Taylor approximation to the definition of the derivative it can be  shown <a name="cite_ref-1"></a>[<sup>[1]</sup>](#cite_note-1) that:
<a name="cite_note-1"></a>

```math
\dot{q} = \lim_{\Delta t \to 0} \frac{q(t + \Delta t) - q(t)}{\Delta t} = \lim_{\Delta t \to 0} \frac{q \otimes \Delta q_L - q}{\Delta t} = \frac{1}{2} [0, \omega] \otimes q=\frac{1}{2} \Omega q
```

where $\otimes$ is the quaternion product operator, and $\Omega$,

```math
\begin{split}\boldsymbol\Omega =
\begin{bmatrix}
0 & -\boldsymbol\omega^T \\ 
\boldsymbol\omega & \lfloor\boldsymbol\omega\rfloor_\times
\end{bmatrix} =
\begin{bmatrix}
0 & -\omega_x & -\omega_y & -\omega_z \\
\omega_x & 0 & \omega_z & -\omega_y \\
\omega_y & -\omega_z & 0 & \omega_x \\
\omega_z & \omega_y & -\omega_x & 0
\end{bmatrix}\end{split}
```


The vector $\lfloor\mathbf{\omega}\rfloor_\times$ is the skew-symmetric matrix representation of the instantanous angular velocity $\mathbf{\omega}$ (Lie algebra vector). It is part of the Lie algebra $\mathfrak{so}(3)$, the space of derivatives of the rotation trajectory that is tangent to $\mathbf{SO(3)}$. The exponential map is used to estimate the rotation that corresponds to a given angular velocity for a certain time duration.

Check this link <a name="cite_ref-3"></a>[link](#cite_note-3) for more info on the above equation. 

Note that this equation assumes that the angular velocity vector is constant over the time period $\Delta t$, and that the orientation change is small. If this is the case in our data then Euler sufficies. For updates be posted over long time periods or with large orientation changes, higher order numerical integration methods may be necessary to solve the equation. 

#### Quaternion state transition
Our state-space system has the form: $\dot{\mathbf q} = \mathbf{Aq}$, where $\mathbf A$ is the system dynamics matrix, i.e. a set of continuous differential equations. We want to find the matrix $\mathbf F$ that propagates the discrete state $\mathbf q$ over the interval $\Delta t$, i.e. $\mathbf q_t = \mathbf F\mathbf q_{t-1}$.

A well known solution to this equation is the matrix exponential. Our discrete state transition for $\Delta t$ is  and can be computed with the following power series:

```math
\mathbf F = e^{\mathbf A\Delta t} = \mathbf{I} + \mathbf A\Delta t  + \frac{(\mathbf A\Delta t)^2}{2!} + \frac{(\mathbf A\Delta t)^3}{3!} + ...
```

substituting $\mathbf{A}$ with the result above, we get:

```math
\hat{\mathbf{q}}_t = \Bigg(\mathbf{I}_4 + \frac{1}{2}\boldsymbol\Omega_t\Delta t +
\frac{1}{2!}\Big(\frac{1}{2}\boldsymbol\Omega_t\Delta t\Big)^2 +
\frac{1}{3!}\Big(\frac{1}{2}\boldsymbol\Omega_t\Delta t\Big)^3 + \cdots\Bigg)\mathbf{q}_{t-1}
```

The series has the known form of the matrix exponential:

```math
e^{\frac{\Delta t}{2}\boldsymbol\Omega(\boldsymbol\omega)} =
\sum_{k=0}^\infty \frac{1}{k!} \Big(\frac{\Delta t}{2}\boldsymbol\Omega(\boldsymbol\omega)\Big)^k
```

By using an $2^{nd}$ order approximation:

```math
\hat{\mathbf{q}}_t =\Big(\mathbf{I}_4 + \frac{\Delta t}{2}\boldsymbol\Omega_t\Big)\mathbf{q}_{t-1}

```

the state transition becomes:

```math
\begin{bmatrix} \hat{q_w} \\ \hat{q_x} \\ \hat{q_y} \\ \hat{q_z}\end{bmatrix} = \begin{bmatrix}
    q_w - \frac{\Delta t}{2} \omega_x q_x - \frac{\Delta t}{2} \omega_y q_y - \frac{\Delta t}{2} \omega_z q_z \\
    q_x + \frac{\Delta t}{2} \omega_x q_w - \frac{\Delta t}{2} \omega_y q_z + \frac{\Delta t}{2} \omega_z q_y \\
    q_y + \frac{\Delta t}{2} \omega_x q_z + \frac{\Delta t}{2} \omega_y q_w - \frac{\Delta t}{2} \omega_z q_x \\
    q_z - \frac{\Delta t}{2} \omega_x q_y + \frac{\Delta t}{2} \omega_y q_x + \frac{\Delta t}{2} \omega_z q_w
\end{bmatrix}

```

The resulting quaternion must be normalized to represent a valid rotation:

```math
\mathbf{q}_{t+1} \gets \frac{\mathbf{q}_{t+1}}{\|\mathbf{q}_{t+1}\|}
```

#### Alternative closed-form solution (previous solution):

Using the Euler-Rodrigues rotation formula we can write the discrete version of the quaternion update as [Sabatini2011]:

```math
\hat{\mathbf{q}}_t =
\Bigg[
\cos\Big(\frac{\|\boldsymbol\omega\|\Delta t}{2}\Big)\mathbf{I}_4 +
\frac{2}{\|\boldsymbol\omega\|}\sin\Big(\frac{\|\boldsymbol\omega\|\Delta t}{2}\Big)\boldsymbol\Omega_t
\Bigg]\mathbf{q}_{t-1}

```

where $\mathbf{I}_4$ is a $4\times 4$ Identity matrix and $\Omega$ as above.

## Kalman Filter

For both position and orientation we track higher order derivatives to capture use cases where the robot stops or moves aggresively. This is more important for the orientation component where we expect the robot to turn aggresirvely when reaching a designated location. Thus we track position $x$, velocity $\dot{x}$, acceleration $\ddot{x}$, quaternion $q$, angular velocity $\omega$, angular acceleration $\dot{\omega}$. Our state vector is: 

```math
\begin{equation}
\mathbf{p} = \begin{bmatrix}
x \ y \ z \ \dot{x} \ \dot{y} \ \dot{z} \ \ddot{x} \ \ddot{y} \ \ddot{z} \ q_{x} \ q_{y} \ q_{z} \ q_{w} \ \omega_{y} \ \omega_{z} \ \omega_{w}  \ \dot{\omega_{y}} \ \dot{\omega_{z}} \ \dot{\omega_{w}}
 \end{bmatrix}^T
\end{equation}

```

where

```math
\boldsymbol{p} = \begin{bmatrix}x \ y \ z\end{bmatrix}
``` 

is the position, 

```math
\boldsymbol{v} = \begin{bmatrix}\dot{x} \ \dot{y} \ \dot{z}\end{bmatrix}
``` 

is the velocity, 

```math
\boldsymbol{a} = \begin{bmatrix}\ddot{x} \ \ddot{y} \ \ddot{z}\end{bmatrix}
``` 

is the acceleration, 

```math
\boldsymbol{q} = \begin{bmatrix}q_{w} \ q_{x} \ q_{y} \ q_{z}\end{bmatrix}
```

the quaternion representation of orientation, 

```math
\boldsymbol{\omega} = \begin{bmatrix}\omega_{x} \ \omega_{y} \ \omega_{z}\end{bmatrix}
```

is the angular velocity vector and 

```math
\boldsymbol{\dot{\omega}} = \begin{bmatrix}\dot{\omega_{x}} \ \dot{\omega_{y}} \ \dot{\omega_{z}}\end{bmatrix}
```

the angular acceleration. In the following we assume a constant acceleration model, i.e. $\boldsymbol{\hat{\ddot{p}}} = \ddot{p}$ and $\boldsymbol{\hat{\dot{\omega}}} = \dot{\omega}$.

### Process model

To find the process model matrix, we need to model the time evolution of each state using Newtonian dynamics and our quaternion update:

```math
\hat{x} = x + \dot{x} \cdot \Delta t + \frac{1}{2} \ddot{x} \cdot \Delta t^2 
```

```math
\hat{y} = y + \dot{y} \cdot \Delta t + \frac{1}{2} \ddot{y} \cdot \Delta t^2 
```

```math
\hat{z} = z + \dot{z} \cdot \Delta t + \frac{1}{2} \ddot{z} \cdot \Delta t^2 
```

```math
\hat{\dot{x}} = \dot{x} + \ddot{x} \cdot \Delta t 
```

```math
\hat{\dot{y}} = \dot{y} + \ddot{y} \cdot \Delta t 
```

```math
\hat{\dot{z}} = \dot{z} + \ddot{z} \cdot \Delta t 
```

```math
\hat{\ddot{x}} = \ddot{x} 
```

```math
\hat{\ddot{y}} = \ddot{y} 
```

```math
\hat{\ddot{z}} = \ddot{z} 
```

```math
\hat{q_w} = q_w - \frac{\Delta t}{2} \omega_x q_x - \frac{\Delta t}{2} \omega_y q_y - \frac{\Delta t}{2} \omega_z q_z 
```

```math
\hat{q_x} = q_x + \frac{\Delta t}{2} \omega_x q_w - \frac{\Delta t}{2} \omega_y q_z + \frac{\Delta t}{2} \omega_z q_y 
```

```math
\hat{q_y} = q_y + \frac{\Delta t}{2} \omega_x q_z + \frac{\Delta t}{2} \omega_y q_w - \frac{\Delta t}{2} \omega_z q_x 
```

```math
\hat{q_z} = q_z - \frac{\Delta t}{2} \omega_x q_y + \frac{\Delta t}{2} \hat{\omega_y} q_x + \frac{\Delta t}{2} \omega_z q_w 
```

```math
\hat{\omega_y} = \omega_y + \dot{\omega_y} \cdot \Delta t 
```

```math
\hat{\omega_z} = \omega_z + \dot{\omega_z} \cdot \Delta t 
```

```math
\hat{\omega_w} = \omega_w + \dot{\omega_w} \cdot \Delta t 
```

```math
\hat{\dot{\omega_y}} = \dot{\omega_x} 
```

```math
\hat{\dot{\omega_z}} = \dot{\omega_y} 
```

```math
\hat{\dot{\omega_w}} = \dot{\omega_z} 
```

### Linearization of the state transition

The Jacobian matrix of the above process model w.r.t the state is the linearization of the state transition model around the current state. We compute it below.

```math
\displaystyle \left[\begin{array}{ccccccccccccccccccc}
1 & 0 & 0 & dt & 0 & 0 & 0.5 dt^{2} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 
0 & 1 & 0 & 0 & dt & 0 & 0 & 0.5 dt^{2} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & dt & 0 & 0 & 0.5 dt^{2} & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 1 & 0 & 0 & dt & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 1 & 0 & 0 & dt & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & dt & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & - \frac{dt w_{xk}}{2} & - \frac{dt w_{yk}}{2} & - \frac{dt w_{zk}}{2} & - \frac{dt q_{xk}}{2} & - \frac{dt q_{yk}}{2} & - \frac{dt q_{zk}}{2} & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \frac{dt w_{xk}}{2} & 1 & \frac{dt w_{zk}}{2} & - \frac{dt w_{yk}}{2} & \frac{dt q_{wk}}{2} & - \frac{dt q_{zk}}{2} & \frac{dt q_{yk}}{2} & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \frac{dt w_{yk}}{2} & - \frac{dt w_{zk}}{2} & 1 & \frac{dt w_{xk}}{2} & \frac{dt q_{zk}}{2} & \frac{dt q_{wk}}{2} & - \frac{dt q_{xk}}{2} & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & \frac{dt w_{zk}}{2} & \frac{dt w_{yk}}{2} & - \frac{dt w_{xk}}{2} & 1 & - \frac{dt q_{yk}}{2} & \frac{dt q_{xk}}{2} & \frac{dt q_{wk}}{2} & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & dt & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & dt & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & dt\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
\end{array}\right]

```

### Measurement model

Our measurements come from either Poser or SLAM. Each measurement is a seven dimensional vector of the estimated state.

```math
\begin{equation}
\mathbf{p_{measure}} = 
\begin{bmatrix}
x_p \ y_p \ z_p \ q_{px} \ q_{py} \ q_{pz} \ q_{pw} 
 \end{bmatrix}^T
\end{equation}

```

The Jacobian of the measurement model is estimated below:

```math
\displaystyle \left[\begin{array}{cccccccccccccccc}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0
\end{array}\right]

```

### State Covariance
Our belief about the state will be designed arround the speed and angular velocity of the platform. We will not explore correlations across dimensions or derivatives. Thus, our state covariance matrix is diagonal:

```math
P =
  \begin{bmatrix}
    \mathcal{\sigma_p} & & & & &\mathbf{[0]_{(n-1) \times n}} \\
    & \mathcal{\sigma_v} & & & \\
    & & \mathcal{\sigma_a} & & \\
    & & & \mathcal{\sigma_q} & \\
    & & & & \mathcal{\sigma_{\omega}} \\
    \mathbf{[0]_{(n-1) \times n}} & & & & & \mathcal{\sigma_{\dot{\omega}}} \\
  \end{bmatrix}

```

We will design our variances to ensure that our measurements are facilitated by our noise model. We know that 98% of the Gaussian occurs within three units of standard deviation. However, the actual data are not Gaussian, and therefore we should allow more values to pass through. So we choose four units of standard deviation. Our variances are setup according to this rule. 

### Measurement Noise Covariance

The observability of yaw in RGB-D SLAM can be limited by the quality of the depth information and the presence of large planar surfaces. Thus, we expect degeneracies in the estimation of the relative pose. Based on this assumption, we initialize our rotation estimate noise to larger values.

```math
  R =
  \begin{bmatrix}
    \sigma_p^2 & \\
    & \sigma_q^2
  \end{bmatrix}

```

We set $\sigma_p^2$ = 1, $\sigma_{qw}^2 = 2$ and $\sigma_{q_{x,y,z}}^2 = 3$. 

### Outlier rejection
pqekf filters outliers using the Mahalanobis distance, to take into account the covariance structure of the distribution. For each measurement, we compute the mahalanobis distance from the predicted measurement based the innovation covariance.

Given a predicted measurement $\boldsymbol{h}$ with mean vector $\boldsymbol{\mu_h}$ and innovation covariance matrix $\boldsymbol{C}$, the Mahalanobis distance $d_M(\boldsymbol{r})$ of an observation vector $\boldsymbol{r}$ is defined as:

```math
 d_M(\boldsymbol{r}) = \sqrt{(\boldsymbol{r} - \boldsymbol{\mu_h})^T \boldsymbol{C}^{-1} (\boldsymbol{r} - \boldsymbol{\mu_h})} 
```

Measurements with $d_M(\boldsymbol{r})$ over some threshold are discarded. 