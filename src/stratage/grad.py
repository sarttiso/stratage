from scipy.optimize import approx_fprime


def finite_differences_loglike(dt_val, params, eps=1e-3):
    def inner_func(params, dt_val):
        return eps_logp(dt_val, params)
    grad_wrt_params = approx_fprime(params, inner_func, eps, dt_val)
    return tuple(grad_wrt_params[:, ii] for ii in range(len(params)))

# define a pytensor Op for our likelihood function


class LogLikeWithGrad(Op):
    def make_node(self, dt_val, params) -> Apply:
        dt_val = pt.as_tensor(dt_val)
        params = pt.as_tensor(params)
        inputs = [dt_val, params]
        outputs = [dt_val.type()]
        return Apply(self, inputs, outputs)

    def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
        # Same as before
        dt_val, params = inputs
        loglike_eval = eps_logp(dt_val, params)
        outputs[0][0] = np.asarray(loglike_eval)

    def grad(
        self, inputs: list[pt.TensorVariable], g: list[pt.TensorVariable]
    ) -> list[pt.TensorVariable]:
        # NEW!
        # the method that calculates the gradients - it actually returns the vector-Jacobian product
        dt_val, params = inputs

        grad_wrt_params = loglikegrad_op(dt_val, params)

        # out_grad is a tensor of gradients of the Op outputs wrt to the function cost
        # print(g)
        [out_grad] = g

        # Compute the vector-Jacobian product for the gradient of params
        params_grad = [pt.sum(out_grad * cur_grad) for cur_grad in grad_wrt_params]

        # Return the placeholder gradient for dt_val (if not differentiating w.r.t. it) and the gradients for params
        return [pytensor.gradient.grad_not_implemented(self, 0, dt_val), pt.stack(params_grad)]


class LogLikeGrad(Op):
    def make_node(self, dt_val, params) -> Apply:
        dt_val = pt.as_tensor(dt_val)
        params = pt.as_tensor(params)
        inputs = [dt_val, params]
        # There are two outputs with the same type as data,
        # for the partial derivatives wrt to m, c
        # print(params.shape)
        outputs = [dt_val.type() for ii in range((n_units + n_contacts))]

        return Apply(self, inputs, outputs)

    def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
        dt_val, params = inputs
        # calculate gradients
        grad_wrt_params = finite_differences_loglike(dt_val, params)

        for ii in range(len(params)):
            outputs[ii][0] = grad_wrt_params[ii]
            # outputs[1][0] = grad_wrt_c


# Initalize the Ops
loglikewithgrad_op = LogLikeWithGrad()
loglikegrad_op = LogLikeGrad()
