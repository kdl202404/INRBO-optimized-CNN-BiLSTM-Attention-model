import numpy as np

def SCNGO(Search_Agents, Max_iterations, Lowerbound, Upperbound, dimensions, objective):
    Lowerbound = np.ones(dimensions) * Lowerbound
    Upperbound = np.ones(dimensions) * Upperbound
    NGO_curve = np.zeros(Max_iterations)

    half = round(Search_Agents / 2)
    X = np.random.rand(half, dimensions) * (Upperbound - Lowerbound) + Lowerbound

    # Reverse learning with fused refraction principle
    k = 0.75
    x_ROBL = np.zeros((half, dimensions))
    for ii in range(half):
        for jj in range(dimensions):
            x_ROBL[ii, jj] = (Lowerbound[jj] + Upperbound[jj]) / 2 + (Lowerbound[jj] + Upperbound[jj]) / 2 * k - X[ii, jj] / k

    X = np.vstack((X, x_ROBL))
    Search_Agents = X.shape[0]

    fit = np.zeros(Search_Agents)
    for i in range(Search_Agents):
        L = X[i, :]
        # Ensure all parameters are within valid range
        L = np.clip(L, Lowerbound, Upperbound)  # Clip parameters to valid range
        fit[i] = objective(L)  # Fitness evaluation

    for t in range(Max_iterations):  # algorithm iteration
        # Update: BEST proposed solution
        best, blocation = np.min(fit), np.argmin(fit)
        if t == 0:
            xbest = X[blocation, :]  # Optimal location
            fbest = best  # The optimization objective function
        elif best < fbest:
            fbest = best
            xbest = X[blocation, :]

        # 正余弦策略变量设置
        ST = 0.5
        ita = 1.2  # ita >= 1
        r1 = (1 - (t / Max_iterations) ** ita) ** (1 / ita)
        Omega = (np.exp(t / Max_iterations) - 1) / (np.exp(1) - 1)

        # UPDATE Northern goshawks based on PHASE 1 and PHASE 2
        X_new = np.zeros_like(X)
        for i in range(Search_Agents):
            # Phase 1: 正余弦策略替换勘探阶段
            r2 = 2 * np.pi * np.random.rand()
            r3 = 2 * np.pi * np.random.rand()
            R2 = np.random.rand(1)

            if R2 < ST:
                for vv in range(dimensions):
                    X_new[i, vv] = Omega * X[i, vv] + r1 * np.sin(r2) * abs(r3 * xbest[vv] - X[i, vv])
            else:
                for vv in range(dimensions):
                    X_new[i, vv] = Omega * X[i, vv] + r1 * np.cos(r2) * abs(r3 * xbest[vv] - X[i, vv])

            # Enforce bounds
            X_new[i, :] = np.clip(X_new[i, :], Lowerbound, Upperbound)

            # Update position based on Eq (5)
            L = X_new[i, :]
            # Ensure parameters are within valid range before evaluation
            L = np.clip(L, Lowerbound, Upperbound)  # Clip parameters to valid range
            fit_new = objective(L)
            if fit_new < fit[i]:
                X[i, :] = X_new[i, :]
                fit[i] = fit_new

            # Phase 2: Exploitation
            R = 0.1 - (np.exp(t / Max_iterations) - 1) / (np.exp(1) - 1)  # 改进R系数
            X_new[i, :] = X[i, :] + (-R + 2 * R * np.random.rand(dimensions)) * X[i, :]

            # Enforce bounds
            X_new[i, :] = np.clip(X_new[i, :], Lowerbound, Upperbound)

            # Update position based on Eq (8)
            L = X_new[i, :]
            fit_new = objective(L)

            # Check for valid parameters
            if fit_new < fit[i]:
                X[i, :] = X_new[i, :]
                fit[i] = fit_new

        # Save best score
        NGO_curve[t] = fbest  # save best solution so far

    return fbest, xbest, NGO_curve
