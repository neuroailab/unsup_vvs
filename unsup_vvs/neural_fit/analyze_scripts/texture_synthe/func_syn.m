function im=func_syn(im0)
    path('/home/chengxuz/matlabPyrTools', path)
    path('/home/chengxuz/textureSynth', path)

    Nsx = 224;  % Synthetic image dimensions
    Nsy = 224;

    Nsc = 3; % Number of pyramid scales
    Nor = 4; % Number of orientations
    Na = 7; % Number of spatial neighbors considered for spatial correlations
    Niter = 25; % Number of iterations of the synthesis loop

    [params] = textureColorAnalysis(im0, Nsc, Nor, Na);
    tic; im = textureColorSynthesis(params, [Nsy Nsx], Niter); toc
end
