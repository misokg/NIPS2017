# A MISO-specific generative model
Suppose that there are two information sources IS 1 and IS 2 that approximate the true objective IS 0. These information sources may each be subject to an unknown bias.  We denote the (unknown) bias of IS 1 by delta1(x) = IS1(x) - IS0(x).  delta2 is defined analogously.

The following animation illustrates how the posterior belief about the internal models of the information sources and the involved biases evolves as we obtain samples from each IS.  In each frame the solid lines give the unknown functions of the models and model discrepancies, whereas the dashed lines show the respective posterior means. The posterior variance is indicated by dots. 
All observations are noiseless.  

Initially, all IS are sampled at the same points. This is the posterior depicted on the second frame. Observe how the posterior of IS 0 is affected from observations at IS 1, IS 2, and IS 0 itself.

<img src="https://github.com/misokg/NIPS2017/blob/master/illustration_misoKG_model.gif" height="530" width="582">
