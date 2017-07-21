% -----------------------------------       
% This script was implemented to plot some of the characteristics of a CHMM.
% First, we load the hmm and the parameters of the four classes of polygons.
% 
% First, we plot the mixture of gaussians calculated for every HMM (group and class)
% and for each state.
% -------------------------------------
        
        load hmmpoligonos2;
       load parampoligonos vlcp;
       % Length of the window to plot the mixture of gaussians 
       longueur_fenetre=499;
       y=cell(nc,ng);
       % Variable to store the mixture of Gaussian
       mixture_of_gaussian=zeros(1,longueur_fenetre+1);
     
      for ic=1:nc
          for ig=1:ng,
              % variables for the loops to store the coefficients of the mixture of gaussian
            Np=length(agrup{ig})-1;
            Ne= length(A{ic,ig}(1,:));
            y {ic,ig}=cell(Np,1);
                for ip=1:Np;
                    y {ic,ig}{ip}=cell(Ne,1);
                    for ie=1:Ne;
                             y {ic,ig}{ip}{ie}=cell(Ngauss{ig}(ip),1);                    
                    for ngauss=1:Ngauss{ig}(ip)
                       % we extract the center of each Gaussians
                      min_mean=  min(Med{ic,ig}{ip}{ie}(:)); % we calculate the minimum of all the gaussians for this state and HMM
                      max_mean=  max(Med{ic,ig}{ip}{ie}(:)); % we calculate the maximum of all the gaussians for this state and HMM
                        % we calculate the maximum variance of the gaussians of all the gaussians for this state and HMM 
                       max_var=  max(Var{ic,ig}{ip}{ie}(:));
                       % We calculate a length to adjust the windows ro plot the mixture of gaussian
                       largeur=abs(max_var*log(0.1));
                       % The path is a variable to estimate the probabilities for "longuer_of_fenetre"+1 points for the gaussian
                       path= (max_mean-min_mean+2*largeur)/longueur_fenetre;
                      % Those points x are the same for all the gaussians from a mixture of gaussian defined by one HMM and one state 
                      x = (min_mean-largeur:path:max_mean+largeur);
%            y {ic,ig}{ip}{ie}{ngauss} is the results of the estimation of the probability for thos points
             y {ic,ig}{ip}{ie}{ngauss}= GAUSSmf(x, [Var{ic,ig}{ip}{ie}(ngauss) Med{ic,ig}{ip}{ie}(ngauss)]);
            % We store the values of the mixture_of_gaussian estimated in those x points 
            % (the results y are multiplied by the coefficient B corresponding to the coefficient of this gaussian for this mixture of gaussian) 
            mixture_of_gaussian=mixture_of_gaussian + B{ic,ig}{ip}{ie}(ngauss) * y {ic,ig}{ip}{ie}{ngauss};
                  end
                  % We plot the mixture of gaussians for each state of this HMM
                  % We make in particular a subplot to show the 12 mixture of gaussians (one for each state) for this HMM and this parameter
                  h3=figure(3);
                            if (ie<6)
                            subplot(2,5,ie)
                             plot(mixture_of_gaussian) ; 
                             mixture_of_gaussian=zeros(1,longueur_fenetre+1);
                                title([' cl',num2str(ic),' gr',num2str(ig),' pr',num2str(ip),' st',num2str(ie)]);
                  
                          else
                                  h4=figure(4);
                                     subplot(2,5,ie-5)
                                    plot(mixture_of_gaussian) ; 
                                mixture_of_gaussian=zeros(1,longueur_fenetre+1);
                                title([' cl',num2str(ic),' gr',num2str(ig),' pr',num2str(ip),' st',num2str(ie)]);
                  
                             end
            end
            plot_dis_symb(ic,ig,ip);
            pause;
           % we could save the figure obtained
          saveas(h3,'fig3.jpg','jpg');
       saveas(h4,'fig4.jpg','jpg');
                
         end 
     end
 end
 
end
