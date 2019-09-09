

def GBM_Returns(Mu, Sigma, dt, Row):
    # Generate returns from a normal distribution
    Mean = (Mu - (Sigma ** 2) / 2) * dt
    Std = Sigma * (dt ** 0.5)
    Return = np.random.normal(Mean, Std, 1)[0]

    return np.exp(Return) - 1




def Power_Utility(Wealth, Gamma):
    if Wealth <= 0:
        return -10

    if Gamma == 1:
        return np.log(Wealth)

    else:
        return (Wealth ** (1 - Gamma)) / (1 - Gamma)
