function [p,val,cpu] = MAM(d,q,M,R,S,mx,theta)
%--------------------------------------------------------------------------
%             Initialization
%--------------------------------------------------------------------------
t0   = tic;
proj_simplex = @(y) max(y-max((cumsum(sort(y,1,'descend'),1)-1)./(1:size(y,1))'),0);
kMax = 10000;
rho  = 10e3;
tol  = 1e-10;
Kn   = round(sqrt(R));

% Initializing theta 
Penal = 0;
if nargin<7
    theta= cell(M,1);
    for m=1:M
       theta{m}  = -d{m}/rho;%+ones(size(d{m}))/R;%zeros(R,S(m));
       Penal     = Penal + norm(d{m}(:))^2;
    end
end
Penal = sqrt(Penal)+1;
a    = (1./S)';
a    = a/sum(a);
pk   = zeros(R,M);
for m=1:M
    pk(:,m) = sum(theta{m},2);
end
p    = zeros(R,1);
paux = p;
val  = 0; 
distB=inf;
%--------------------------------------------------------------------------
%          Main Loop
%--------------------------------------------------------------------------
kPrint = 1;
for k=1:kMax
    % Print some information at every 10 iterations
    if (mod(k, kPrint) == 0)
         fprintf(1,'k = %5.0f, distB = %5.2e, val = %8.4f, valPen = %8.4f, cpu = %5.0f \n',k,distB,mx*val,mx*(val+Penal*distB),toc(t0)); 
         imagesc(reshape(1-p,Kn,Kn));
         title(strcat('k=',num2str(k),', val=',num2str(mx*val),', distB=', num2str(distB),', t=',num2str(round(toc(t0)))));
         colormap hot;
         name = strcat('MAM\',num2str(k),'.png'); 
         colormap hot;
         %salvaPNG(gcf,name)
         %save(strcat('MAM\',num2str(k),'.mat'),"p");         
         pause(0.01) % Pause cafe
        if distB<=tol 
            break
        end
    end
    %
    p = pk*a;
    if (mod(k+1, kPrint) == 0)
        val = 0;
        for m=1:M
            pihat    = proj_simplex( (theta{m} + 2*(p-pk(:,m))/S(m) - (1/rho)*d{m})./q{m}' ).*q{m}' ;
            theta{m} = pihat - (p-pk(:,m))/S(m);
            val      = val + sum(d{m}.*pihat,'all');
            paux(:,m)= sum(pihat,2);
            pk(:,m)  = sum(theta{m},2);
        end
        p     = paux*a;
        distB = 0;
        for m=1:M
            distB = distB + (norm(p-paux(:,m)).^2)/S(m);
        end
        distB = sqrt(distB);
    else
        for m=1:M         
            theta{m} = proj_simplex( (theta{m} + 2*(p-pk(:,m))/S(m) - (1/rho)*d{m})./q{m}' ).*q{m}' - (p-pk(:,m))/S(m);
            pk(:,m)  = sum(theta{m},2);
        end         
    end
end
%--------------------------------------------------------------------------
%   Compute the function value before going out
%--------------------------------------------------------------------------
p   = max(pk*a,0);
val = 0;
for m=1:M
    val = val + sum(d{m}.*proj_simplex( (theta{m} + 2*(p-pk(:,m))/S(m) - (1/rho)*d{m})./q{m}' ).*q{m}','all');
end
val = mx*val;
cpu = toc(t0);
fprintf(1,'k = %5.0f, distB = %5.2e, val = %8.4f, cpu = %5.0f \n',k,distB,mx*val,toc(t0)); 
return
