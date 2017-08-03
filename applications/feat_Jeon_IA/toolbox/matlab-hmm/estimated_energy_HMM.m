function   [ener_hi,prev_alpha]=estimated_energy_HMM(frm_indx,states,nbStates,pow,prev_alpha,idx1,SB,model)
      load(model);
        if frm_indx==1
           b=zeros(states,1);
           for i =1:states 
               distance=zeros(states,nbStates);
               for j=1:nbStates
                    distance(i,j)=distance(j)+mixmat1(i,j)*gaussPDF(pow(1:idx1*2-1)',mu1(1:idx1*2-1,i,j),sigma1(1:idx1*2-1,1:idx1*2-1,i,j));
               end
               b(i)=sum(distance(i,:));
           end
%            b=b/sum(b);
           alpha=prior1.*b;
           prev_alpha=alpha;
        else
           b=zeros(states,1);
           for i =1:states 
               distance=zeros(states,nbStates);
               for j=1:nbStates
                    distance(i,j)=distance(i,j)+mixmat1(i,j)*gaussPDF(pow(1:idx1*2-1)',mu1(1:idx1*2-1,i,j),sigma1(1:idx1*2-1,1:idx1*2-1,i,j));
                    gauss(j) = gaussPDF(pow(1:idx1*2-1)',mu1(1:idx1*2-1,i,j),sigma1(1:idx1*2-1,1:idx1*2-1,i,j));
               end
               b(i)=sum(distance(i,:));
           end
%            if sum(b)==0
%                b = zeros(states,1);
%                b(1:end) = 1/states;
%            else
%                b=b/sum(b);
%            end

           alpha_tmp=prev_alpha'*transmat1;
           alpha=(alpha_tmp)'.*b;
           prev_alpha=alpha/sum(alpha);

        end

        ener_hi=zeros(states,2*SB+1-(idx1*2+1));   % 의심되는 부분

        for i =1:states 
            for j=1: nbStates
                distance(i,j)=distance(i,j)+mixmat1(i,j)*gaussPDF(pow(1:idx1*2-1)',mu1(1:idx1*2-1,i,j),sigma1(1:idx1*2-1,1:idx1*2-1,i,j));%*prob(k);
            end
            total_prob=sum(distance(i,:));
            gmm_prob= distance(i,:)/(total_prob);
            for j=1:nbStates
                ener_hi(i,:)=ener_hi(i,:)+gmm_prob(j).*mu1(idx1*2:2*SB-1,i,j)';%*G{1}.mu(2*SB:end,num1); %mu1(1:2*SB-1,i,j)
            end
            % 예외처리 %
            if sum(distance(i,:))==0
                ener_hi(i,:)=0;
                for num1=1:nbStates
                    w=dist(pow',mu1(1:idx1*2-1,i,num1));
                    ener_hi(i,:)=ener_hi(i,:)+mu1(idx1*2:2*SB-1,i,num1)'*(1/w);
                end
            end

        end 
        alpha_ratio=alpha/sum(alpha);

        for i =1:states 
            ener_hi(i,:)=ener_hi(i,:).*alpha_ratio(i);
        end
        if states>1
            ener_hi=sum(ener_hi);
        end
end        