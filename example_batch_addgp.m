function example_batch_addgp(iter,rep, bch_size, high_dim, test_func,bo_method)
    % Copyright (c) 2017 Zi Wang, Chengtao Li

    % add necessary paths
    deploy;
    addpath('utils/dpp/helpers')
    addpath('utils/dpp/sampling')

    % Define function
    if strcmp(test_func,'WalkerSpeed')
        dx = high_dim;
        xmin = zeros(dx,1)-2;
        xmin(1)=1;
        xmax = zeros(dx,1)+2;
        xmax(1)=10;
        f = @(x) walker_speed(x);
    elseif strcmp(test_func,'Branin')
        dx = high_dim;
        xmin = zeros(dx,1);
        xmin(1)=-5;
        xmin(2)=0;
        xmax = zeros(dx,1);
        xmax(1)=10;
        xmax(2)=15;
        f=@(x) -(branin(x));
    elseif strcmp(test_func,'Hartmann6')
        dx = high_dim;
        xmin = zeros(dx,1);
        xmax = ones(dx,1);
        f=@(x) -(hartmann(x));
    elseif strcmp(test_func,'Rosenbrock')
        dx = high_dim;
        xmin = zeros(dx,1)-2;
        xmax = zeros(dx,1)+2;
        f=@(x) -(rosenbrock(x));
    end
    
    % f = sample_addGP(dx, dx, xmin, xmax);
    % f=@(x) sum(-x.^2);
    % Save the file to a path 
    options.savefilenm = 'tmp.mat';

    % Set the GP hyper-parameters if you would like to fix them.
    % Comment the following 3 lines out if you would like to learn them.
%     options.l = ones(1,dx)*50;
%     options.sigma = ones(1, dx)*5;
%     options.sigma0 = 0.0001*ones(1, dx);

    % bo_method chosen from 
    % 'batch_rand', {'batch_ucb_dpp', 'batch_ucb_pe'} * {'_ori', '_fnc'}
    options.bo_method = bo_method;
    % batch_size
    options.batch_size = bch_size;
    % split number for each dimension
    options.num_split = 10;
    % set the seed for random numbers
    options.seed=rep;
    % generate the grid of indices first
    gen_grids(options.num_split);

    % Start BO with batch add-GP
    % The additive learning strategy is based on the paper
    % Wang, Zi and Li, Chengtao and Jegelka, Stefanie and Kohli, Pushmeet. 
    % Batched High-dimensional Bayesian Optimization via Structural Kernel 
    % Learning. arXiv preprint arXiv:1703.01973.
    file_name=strcat('Results-add/SKL_',test_func,'_',bo_method,'_BtchSize',num2str(bch_size),'_Dim',num2str(high_dim),'_rep',num2str(rep));
    [max_fs, cons_time]=batch_add_gpopt(f, xmin, xmax, iter , [], [], options, file_name);
    