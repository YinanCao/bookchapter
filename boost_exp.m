clear; clc; close all

Iax   = linspace(130,182,5); %  values of target A
Ibx   = 156;                 %  value  of target B (fixed)
Idx   = [26,104];            %  values of distactor D
pow   = 1;
Iax   = Iax.^pow;
Ibx   = Ibx.^pow;
Idx   = Idx.^pow;
nD    = numel(Idx);
K     = 100; % gain
semi  = 50;  % semi-saturation
sigma = [50,6,12]; % decision noise
w     = 1;  % normalization weight
scale = 0;  % mean-dependent noise
boost = [1,3,5]; % multiplicative boost factor
nsim  = 1e3;     % n simulated samples

for model = 3:-1:1
    for condition = 1:numel(boost)
        c = boost(condition);
        for i = 1:numel(Iax)
            for k = 1:numel(Idx)
                % boost:
                A = Iax(i)*c; 
                B = Ibx*c;
                D = Idx(k)*c;

                switch model
                    case 1
                    % Baseline model:
                    % normalization weight = 0
                    mu1 = K*A/semi; 
                    mu2 = K*B/semi;
                    mu3 = K*D/semi;
                    case 2
                    % Divisive-norm model:
                    mu1 = K*(A./((semi+(A+B+D)*w)));
                    mu2 = K*(B./((semi+(A+B+D)*w)));
                    mu3 = K*(D./((semi+(A+B+D)*w)));
                    case 3
                    % Range-norm model:
                    range = max([A,B,D]) - min([A,B,D]);
                    mu1 = K*((A-min([A,B,D]))./((semi+(range)*w)));
                    mu2 = K*((B-min([A,B,D]))./((semi+(range)*w)));
                    mu3 = K*((D-min([A,B,D]))./((semi+(range)*w)));
                end

                %%% analytical solution:
                sd_possion = sqrt(scale.*[mu1,mu2,mu3]);
                sd = sqrt(sd_possion.^2 + sigma(model).^2);
                x = -5000:1:5000;
                kk = [trapz(x,(normpdf(x,mu1,sd(1)).*(normcdf(x,mu2,sd(2))).*(normcdf(x,mu3,sd(3)))))
                      trapz(x,(normpdf(x,mu2,sd(2)).*(normcdf(x,mu1,sd(1))).*(normcdf(x,mu3,sd(3)))))
                      trapz(x,(normpdf(x,mu3,sd(3)).*(normcdf(x,mu1,sd(1))).*(normcdf(x,mu2,sd(2)))))
                      ];
                % conditional p(A|A,B):
                pA_AB_ana{model}(i,k,condition) = kk(1)/sum(kk(1:2));

                %%% simulated solution:
                x1 = mu1 + randn(nsim,1).*sqrt(scale.*mu1) + randn(nsim,1).*sigma(model);
                x2 = mu2 + randn(nsim,1).*sqrt(scale.*mu2) + randn(nsim,1).*sigma(model);
                x3 = mu3 + randn(nsim,1).*sqrt(scale.*mu3) + randn(nsim,1).*sigma(model);
                [~,opt] = max([x1,x2,x3],[],2);
                Na = numel(find(opt==1));
                Nb = numel(find(opt==2));
                pA_AB_sim{model}(i,k,condition) = Na/(Na+Nb);
            end % D
        end % A
    end % boost
end % model

close all; clc

pA_AB   = pA_AB_ana; % use analytical solution
Nfit    = 6;
x       = -135:135; % curve x-corr
fs      = 15;       
alpha   = [0.2, 1]; % distractor level, alpha level for plots
ccc     = {[205,112,107], [96,151,200], [29,177,80]};
nmodel  = numel(pA_AB);
tau_out = [];
for model = nmodel:-1:1
    subplot(2,nmodel,model)
    for condition = 1:numel(boost) % boost factor
        c = boost(condition);
        for k = 1:numel(Idx) % distractor
            model_x = (Iax - Ibx)'*c;
            model_y = pA_AB{model}(:,k,condition); % simulated data
            tau     = fitGauss(model_x,model_y,Nfit)*sqrt(2);
            y       = 0.5 + 0.5 * erf(x/tau);
            patch([x fliplr(x)],[y fliplr(y)],ccc{condition}/255,...
                'EdgeColor',ccc{condition}/255,...
                'LineWidth',2,'EdgeAlpha',alpha(k)); hold on;
            plot(model_x,model_y,'o','markersize',4,'color',ccc{condition}/255)
            % axis square
            set(gca,'box','off','color','none','fontsize',fs,'linewidth',1)
            set(gca,'tickdir','out')
            xlim([min(x),max(x)])
            tau_out(condition,k,model) = tau;
        end
    end
    if model==1
        ylabel('p(A | A, B)')
        xlabel('A - B')
    end
end
set(gcf,'renderer','painters')

for model = nmodel:-1:1
    subplot(2,nmodel,model+nmodel)
    tau = tau_out(:,:,model);
    b = bar(tau, 'grouped');
    for k = 1:size(tau, 2) % Iterate over the two bars in each group
        b(k).FaceColor = 'flat'; % Allow individual bar coloring
        b(k).EdgeColor = 'flat'; % Allow individual edge coloring
        for i = 1:size(tau, 1) % Iterate over the three groups
            % Assign colors
            b(k).CData(i, :) = ccc{i}/255;
            b(k).BaseValue = 1;
            % Assign alpha transparency
            b(k).FaceAlpha = alpha(k);
        end
    end
    ylim([0,200])
    set(gca,'box','off','color','none','fontsize',fs,'linewidth',1)
    set(gca,'tickdir','out')
    if model==1
        ylabel('cdf \sigma')
        xlabel('Multiplicative factor')
    end
    set(gca,'xticklabel',boost)
    % set(gcf,'renderer','opengl')
end

figure('position',[446   534   365   413])
I = [Iax,Idx];
for k = 1:numel(boost)
    plot((1:numel(I))+numel(I)*(k-1),sort(I*boost(k)),'o','color',ccc{k}/255,...
        'markerfacecolor',ccc{k}/255,'markersize',10)
    hold on;
    axis tight
    set(gca,'box','off','color','none','fontsize',fs,'linewidth',1)
    set(gca,'tickdir','out','ytick',[100,400,800])
    xlabel('Multiplicative factor')
end


%%
function Xfit = fitGauss(x,y,Nfit)

    obFunc = @(b) LL_fun(x,y,b);

    b     = [1e-5,1e3];
    B     = b;
    B_lim = b;
    LB    = B_lim(:,1); 
    UB    = B_lim(:,2);

    % grid search starting points:
    Nall = Nfit*2;
    X0 = zeros(Nall,size(B,1));
    for i = 1:size(B,1)
        a = B(i,1); 
        b = B(i,2);
        X0(:,i) = a + (b-a).*rand(Nall,1);
    end
    % Np = sum(std(X0)~=0); % number of free parameters
    feval = [1e4, 1e4]; % max number of function evaluations and iterations
    options = optimset('MaxFunEvals',feval(1),'MaxIter',feval(2),...
        'TolFun',1e-20,'TolX',1e-20,'Display','none');

    X0_valid = [];
    for iter = 1:Nall
        init_fval = obFunc(X0(iter,:));
        if isreal(init_fval) && ~isnan(init_fval) && ~isinf(init_fval)
            X0_valid = [X0_valid; X0(iter,:)];
        end
    end

    tic
    for iter = Nfit:-1:1
        [Xfit_grid(iter,:), NegLL_grid(iter)] = fmincon(obFunc,...
            X0_valid(iter,:),[],[],[],[],LB,UB,[],options);
    end
    toc

    [~,best] = min(NegLL_grid);
    Xfit = Xfit_grid(best,:);

end

function error = LL_fun(x,p,sigma)
    tau = sqrt(2)*sigma;
    p_pred = 0.5 + 0.5 * erf(x/tau);
    p(p==0) = 1e-6;
    p(p==1) = 1-1e-6;
    error = -sum(p.*log(p_pred) + (1-p).*log(1-p_pred));
end






