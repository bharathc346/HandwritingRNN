def log_likelihood(end, log_pi, mu, log_sigma, rho, y, masks):
    # targets
    y_0 = y.narrow(-1,0,1)
    y_1 = y.narrow(-1,1,1)
    y_2 = y.narrow(-1,2,1)

    mu_1, mu_2 = torch.split(mu, 20, dim=-1)
    log_sigma_1, log_sigma_2 = torch.split(log_sigma, 20, dim=-1)

    # end of stroke prediction
    end_loglik = (y_0*end + (1-y_0)*(1-end)).log().squeeze()

    # new stroke point prediction
    const = 1E-20 # to prevent numerical error
    pi_term = torch.Tensor([2*np.pi]).to(DEVICE)
    if cuda:
        pi_term = pi_term.cuda()
    pi_term = -Variable(pi_term, requires_grad = False).log()

    z = (y_1 - mu_1)**2/(log_sigma_1.exp()**2)\
        + ((y_2 - mu_2)**2/(log_sigma_2.exp()**2)) \
        - 2*rho*(y_1-mu_1)*(y_2-mu_2)/((log_sigma_1 + log_sigma_2).exp())
    mog_lik1 =  pi_term -log_sigma_1 - log_sigma_2 - 0.5*((1-rho**2).log())
    mog_lik2 = z/(2*(1-rho**2))
    mog_loglik = ((log_pi + (mog_lik1 - mog_lik2)).exp().sum(dim=-1)+const).log()

    return (end_loglik*masks).sum() + ((mog_loglik)*masks).sum()

def compute_nll_loss(end, log_pi, mu, log_sigma, rho, targets, masks, M=20):
    epsilon = 1e-6

    mu_1, mu_2 = torch.split(mu, M, dim=-1)
    log_sigma_1, log_sigma_2 = torch.split(log_sigma, M, dim=-1)

    log_constant = log_pi - math.log(2 * math.pi) - log_sigma_1 - log_sigma_2 - 0.5 * torch.log(epsilon + 1 - rho.pow(2))

    x1 = targets[:, :, 1:2]
    x2 = targets[:, :, 2:]

    std_1 = torch.exp(log_sigma_1) + epsilon
    std_2 = torch.exp(log_sigma_2) + epsilon

    X1 = ((x1 - mu_1) / std_1).pow(2)
    X2 = ((x2 - mu_2) / std_2).pow(2)
    X1_X2 = 2 * rho * (x1 - mu_1) * (x2 - mu_2) / (std_1 * std_2)

    Z = X1 + X2 - X1_X2

    X = -Z / (2 * (epsilon + 1 - rho.pow(2)))

    log_sum_exp = torch.logsumexp(log_constant + X, 2)
    BCE = nn.BCEWithLogitsLoss(reduction='none')

    loss_t = -log_sum_exp + BCE(end, targets[:, :, 0])
    loss = torch.sum(loss_t * masks)

    return loss
