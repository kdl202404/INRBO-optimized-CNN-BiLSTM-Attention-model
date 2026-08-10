import numpy as np

def boundary_check(X, LB, UB):
    return np.clip(X, LB, UB)

# def initialization(N, dim, UB, LB):
#     return np.random.uniform(LB, UB, (N, dim))

#使用混沌序列进行初始化,混沌序列能够增强初始种群的多样性，避免算法陷入局部最优。
def chaotic_initialization(N, dim, UB, LB, d=0.7):
    # 使用Tent混沌映射来初始化
    X = np.random.rand(N, dim)
    chaotic_X = np.where(X < d, X / d, (1 - X) / (1 - d))
    return LB + chaotic_X * (UB - LB)

#自适应惯性权重调整,为了提升算法的探索和开发能力，可以引入动态调整的惯性权重 ω，使其在迭代初期偏向探索，后期偏向开发
def adaptive_weight(it, MaxIt, omega_max=0.7, omega_min=0.4, alpha=10, beta=0.35):
    return omega_min + (omega_max - omega_min) / (1 + np.exp(-alpha * (it / MaxIt - beta)))



#可以在陷入局部最优时使用混沌扰动（如Logistic映射）来更新解的位置，增强算法跳出局部最优的能力。
def chaotic_perturbation(X, LB, UB):
    r = 4  # Logistic映射参数
    chaos = np.random.rand(X.shape[0])
    for _ in range(10):  # 迭代10次以增强混沌效果
        chaos = r * chaos * (1 - chaos)
    perturbation = LB + chaos * (UB - LB)
    return np.clip(X + perturbation, LB, UB)



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

    # Position = initialization(N, dim, UB, LB)
    Position = chaotic_initialization(N, dim, UB, LB)

    Fitness = np.zeros(N)

    for i in range(N):
        Fitness[i] = fobj(Position[i, :])

    Ind = np.argsort(Fitness)
    Best_Score = Fitness[Ind[0]]
    Best_Pos = Position[Ind[0], :]
    Worst_Cost = Fitness[Ind[-1]]
    Worst_Pos = Position[Ind[-1], :]

    CG_curve = np.zeros(MaxIt)


    # 设置局部最优检测的参数
    stagnation_counter = 0  # 计数器
    stagnation_limit = 10   # 局部最优检测的次数限制
    improvement_threshold = 1e-6  # 改进阈值



    for it in range(MaxIt):
        # delta = (1 - ((2 * it) / MaxIt)) ** 5

        #这会根据迭代次数自动调整权重，有助于提高搜索效率。
        delta = adaptive_weight(it, MaxIt)

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

            # if Xnew_Cost < Fitness[i]:
            #     Position[i, :] = Xnew
            #     Fitness[i] = Xnew_Cost
            #
            #     if Fitness[i] < Best_Score:
            #         Best_Pos = Position[i, :]
            #         Best_Score = Fitness[i]
            #
            # if Fitness[i] > Worst_Cost:
            #     Worst_Pos = Position[i, :]
            #     Worst_Cost = Fitness[i]

            # 检查是否更新最优解
            if Xnew_Cost < Fitness[i]:
                Position[i, :] = Xnew
                Fitness[i] = Xnew_Cost

                if Fitness[i] < Best_Score:
                    if Best_Score - Fitness[i] < improvement_threshold:
                        stagnation_counter += 1
                    else:
                        stagnation_counter = 0  # 重置计数器

                    Best_Pos = Position[i, :]
                    Best_Score = Fitness[i]

            # 更新最差解
            if Fitness[i] > Worst_Cost:
                Worst_Pos = Position[i, :]
                Worst_Cost = Fitness[i]

            # 检查是否陷入局部最优，并应用混沌扰动
            if stagnation_counter >= stagnation_limit:
                Position[i, :] = chaotic_perturbation(Position[i, :], LB, UB)
                stagnation_counter = 0  # 重置计数器

        CG_curve[it] = Best_Score

    return Best_Score, Best_Pos, CG_curve
