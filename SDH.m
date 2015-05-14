function [G, F, B] = SDH(X,y,B,gmap,Fmap,tol,maxItr,debug)

% ---------- Argument defaults ----------
if ~exist('debug','var') || isempty(debug)
    debug=1;
end
if ~exist('tol','var') || isempty(tol)
    tol=1e-5;
end
if ~exist('maxItr','var') || isempty(maxItr)
    maxItr=1000;
end
nu = Fmap.nu;
delta = 1/nu;
% ---------- End ----------

% label matrix N x c
if isvector(y) 
    Y = sparse(1:length(y), double(y), 1); Y = full(Y);
else
    Y = y;
end


% G-step
switch gmap.loss
    case 'L2'
        [Wg, ~, ~] = RRC(B, Y, gmap.lambda); % (Z'*Z + gmap.lambda*eye(nbits))\Z'*Y;
    case 'Hinge'
        svm_option = ['-q -s 4 -c ', num2str(1/gmap.lambda)];
        model = train(double(y),sparse(B),svm_option);
        Wg = model.w';
end
G.W = Wg;

% F-step

[WF, ~, ~] = RRC(X, B, Fmap.lambda);

F.W = WF; F.nu = nu;


i = 0; 
while i < maxItr    
    i=i+1;  
    
    if debug,fprintf('Iteration  %03d: ',i);end
    
    % B-step
  
        XF = X*WF;
   
    switch gmap.loss
        case 'L2'
            Q = nu*XF + Y*Wg';
            
            B = zeros(size(B));          
            for time = 1:10           
               Z0 = B;
                for k = 1 : size(B,2)
                    Zk = B; Zk(:,k) = [];
                    Wkk = Wg(k,:); Wk = Wg; Wk(k,:) = [];                    
                    B(:,k) = sign(Q(:,k) -  Zk*Wk*Wkk');
                end
                
                if norm(B-Z0,'fro') < 1e-6 * norm(Z0,'fro')
                    break
                end
            end
        case 'Hinge' 
            
            for ix_z = 1 : size(B,1)
                w_ix_z = bsxfun(@minus, Wg(:,y(ix_z)), Wg);
                B(ix_z,:) = sign(2*nu*XF(ix_z,:) + delta*sum(w_ix_z,2)');
            end
             
    end

    
    % G-step
    switch gmap.loss
    case 'L2'
        [Wg, ~, ~] = RRC(B, Y, gmap.lambda); % (Z'*Z + gmap.lambda*eye(nbits))\Z'*Y;
    case 'Hinge'        
        model = train(double(y),sparse(B),svm_option);
        Wg = model.w';
    end
    G.W = Wg;
    
    % F-step 
    WF0 = WF;
    
    [WF, ~, ~] = RRC(X, B, Fmap.lambda);
   
    F.W = WF; F.nu = nu;
    
    
    
    
    bias = norm(B-X*WF,'fro');
    
    if debug, fprintf('  bias=%g\n',bias); end
    
    if bias < tol*norm(B,'fro')
            break;
    end 
    
    
    if norm(WF-WF0,'fro') < tol * norm(WF0)
        break;
    end
    
    
end