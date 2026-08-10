import numpy as np

def boundary_check(X, LB, UB):
    return np.clip(X, LB, UB)

def initialization(N, dim, UB, LB):
    return np.random.uniform(LB, UB, (N, dim))

def search_rule(Best_Pos, Worst_Pos, Position, rho, Flag):
    dim = Position.shape[0]
    DelX = np.random.rand(dim) * np.abs(Best_Pos - Position)

    denominator = 2 * (Best_Pos + Worst_Pos - 2 * Position) + 1e-10  # 避免除以零
    NRSR = np.random.randn() * ((Best_Pos - Worst_Pos) * DelX) / denominator

    if Flag == 1:
        Xa = Position - NRSR + rho
    else:
        Xa = Best_Pos - NRSR + rho

    r1, r2 = np.random.rand(2)
    yp = r1 * (np.mean(Xa + Position) + r1 * DelX)
    yq = r2 * (np.mean(Xa + Position) - r2 * DelX)
    denominator = 2 * (yp + yq - 2 * Position) + 1e-10  # 避免除以零
    NRSR = np.random.randn() * ((yp - yq) * DelX) / denominator

    return NRSR


def NRBO(N, MaxIt, LB, UB, dim, fobj):
    DF = 0.6
    LB = np.ones(dim) * LB
    UB = np.ones(dim) * UB

    Position = initialization(N, dim, UB, LB)
    Fitness = np.zeros(N)

    for i in range(N):
        Fitness[i] = fobj(Position[i, :])

    Ind = np.argsort(Fitness)
    Best_Score = Fitness[Ind[0]]
    Best_Pos = Position[Ind[0], :]
    Worst_Cost = Fitness[Ind[-1]]
    Worst_Pos = Position[Ind[-1], :]

    CG_curve = np.zeros(MaxIt)

    for it in range(MaxIt):
        delta = (1 - ((2 * it) / MaxIt)) ** 5

        for i in range(N):
            P1 = np.random.choice(N, 2, replace=False)
            a1, a2 = P1[0], P1[1]

            rho = np.random.rand() * (Best_Pos - Position[i, :]) + np.random.rand() * (Position[a1, :] - Position[a2, :])

            Flag = 1
            NRSR = search_rule(Best_Pos, Worst_Pos, Position[i, :], rho, Flag)
            X1 = Position[i, :] - NRSR + rho
            X2 = Best_Pos - NRSR + rho

            Xupdate = np.zeros(dim)
            for j in range(dim):
                X3 = Position[i, j] - delta * (X2[j] - X1[j])
                a1, a2 = np.random.rand(), np.random.rand()
                Xupdate[j] = a1 * (a1 * X1[j] + (1 - a2) * X2[j]) + (1 - a2) * X3

            if np.random.rand() < DF:
                theta1 = -1 + 2 * np.random.rand()
                theta2 = -0.5 + np.random.rand()
                beta = np.random.rand() < 0.5
                u1 = beta * 3 * np.random.rand() + (1 - beta)
                u2 = beta * np.random.rand() + (1 - beta)

                if u1 < 0.5:
                    X_TAO = Xupdate + theta1 * (u1 * Best_Pos - u2 * Position[i, :]) + theta2 * delta * (u1 * np.mean(Position, axis=0) - u2 * Position[i, :])
                else:
                    X_TAO = Best_Pos + theta1 * (u1 * Best_Pos - u2 * Position[i, :]) + theta2 * delta * (u1 * np.mean(Position, axis=0) - u2 * Position[i, :])

                Xnew = X_TAO
            else:
                Xnew = Xupdate

            Xnew = boundary_check(Xnew, LB, UB)
            Xnew_Cost = fobj(Xnew)

            if Xnew_Cost < Fitness[i]:
                Position[i, :] = Xnew
                Fitness[i] = Xnew_Cost

                if Fitness[i] < Best_Score:
                    Best_Pos = Position[i, :]
                    Best_Score = Fitness[i]

            if Fitness[i] > Worst_Cost:
                Worst_Pos = Position[i, :]
                Worst_Cost = Fitness[i]

        CG_curve[it] = Best_Score

    return Best_Score, Best_Pos, CG_curve
