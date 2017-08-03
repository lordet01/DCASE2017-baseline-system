function feats = make_feat_mfcc_rastamat( signal, fs, paramType, params )

% feat = make_feat_mfcc_rastamat( sig, fs, paramType )
% feat = make_feat_mfcc_rastamat( sig, fs, 1 ) %'mel'
% feat = make_feat_mfcc_rastamat( sig, fs, 2 ) % 'linear' 
%
if( nargin < 4 )
    params.version = 2 ;  % using the revised mfcc fucntion
    params.numcep = 20 ;
    params.dither = 1 ;
    params.prefilter = 1 ; 
    params.minfreq = 300 ; %
    params.maxfreq = 3400 ; %
    params.wintime = 0.02 ; % 20 ms
    params.nbands = 32 ; % 32 is the number that the linconln lab system used.
    params.modelorder = 0 ; %assuming mfcc , not plp
    params.C0 = 0 ;
    params.delta = 1 ;
   params.delta_half_win = 5 ;   % 5 was obtaiend by using sre08
%    decimated subset and run difrent window size 
%    params.delta_half_win = 2 ; % default one in LL 
    params.compression = 'log' ;
    params.sig_norm    = 1 ; 
    params.usecmp = 0 ;
    params.invert_melscale = 0 ; 
    params.dorasta = 0 ; 
end


if( paramType == 1 )
     params.brkfrq = 1000 ; % default for mfcc
     params.dorasta = 0 ; 
end

if( paramType == 2 )
     params.brkfrq = 4000 ; % to get linear scale
     params.dorasta = 0 ; 
end

if( paramType == 3 )
     params.brkfrq = 1000 ; % default for mfcc, with rasta
     params.dorasta = 1 ; 
end

if( paramType == 4 )
     params.brkfrq = 4000 ; % to get linear scale, , with rasta
     params.dorasta = 1 ; 
end

%
sr = fs ; 
if(params.prefilter == 1)
%     [b, a] = potsband( sr, 300, 3140 ) ; params.minfreq
    [b, a] = potsband( sr, params.minfreq, params.maxfreq ) ; 
    signal = filter( b, a, signal ) ;
end
%
%         if(params<=4)
%             wintime = 0.02 ; nbands = 25 ;
%         else
%             wintime = 0.025 ; nbands = 25 ; params = params - 4 ;
%         end
%         params = params + 4 ;
%         wintime = 0.02 ; % nbands = 32 ;
%         [C, AS, PS]    = melfcc(signal, sr, 'numcep', 20, 'dither', 1, 'minfreq', 300, 'maxfreq', 3140 , 'wintime', params.wintime, 'nbands', nbands);
if( strcmpi( params.compression, 'log' ))
    fprintf( 1, 'log compression in melfcc \n' ) ;
    if( params.sig_norm == 1)
        fprintf( 1, 'normalization in log compression in melfcc \n' ) ;
        signal =  unitseq( signal) ;
    end
    
    if(params.version==1)
        fprintf( 1, 'melfcc  original melscale\n' ) ;
        [C, AS, PS]    = melfcc(signal, sr, 'numcep', params.numcep, 'dither', params.dither, ...
            'minfreq', params.minfreq, 'maxfreq', params.maxfreq , ...
            'wintime', params.wintime, 'nbands',  params.nbands,  'modelorder', params.modelorder, 'usecmp', params.usecmp);
    end
    
    if(params.version==2)
        fprintf( 1, 'melfcc  quasi melscale\n' ) ;
        [C, AS, PS]    = melfcc_Xinhui(signal, sr,  params,  'numcep', params.numcep, 'dither', params.dither, ...
            'minfreq', params.minfreq, 'maxfreq', params.maxfreq , ...
            'wintime', params.wintime, 'nbands',  params.nbands,  'modelorder', params.modelorder, 'usecmp', params.usecmp);
    end
    
elseif (strcmpi( params.compression, 'cubic' ))
    fprintf( 1, 'cubic compression in melfcc \n' ) ;
    if( params.sig_norm == 1)
        fprintf( 1, 'normalization in cubic compression in melfcc \n' ) ;
        
        signal =  unitseq( signal ) ;
    end
    [C, AS, PS]    = melfcc_cubic(signal, sr, 'numcep', params.numcep, 'dither', params.dither, ...
        'minfreq', params.minfreq, 'maxfreq', params.maxfreq , ...
        'wintime', params.wintime, 'nbands',  params.nbands,  'modelorder', params.modelorder, 'usecmp', params.usecmp);
end
%feats1 = C(2:20, :);
if( params.C0 == 0)
    feats1 = C(2:params.numcep, :); %
else
    feats1 = C(1:params.numcep, :); % C0 added
end

if( params.delta == 0 )
    feats = [feats1];
elseif( params.delta == 1 )
    d1 = deltas(feats1, 2*params.delta_half_win+1);
    feats = [feats1; d1];
elseif( params.delta == 2 )
    d1 = deltas(feats1, 2*params.delta_half_win+1);
    d2 = deltas(d1    , 2*params.delta_half_win+1);
    feats = [feats1; d1; d2];
end



function x = unitseq(x)
% UNITSEQ sequence normalization to N(0, 1)
%	x = unitseq(x);
%	UNITSEQ makes a sequence to be a zero-mean sequence 
%	with variance 1 (i.e., N(0, 1)).

% Auther: Powen Ru (powen@isr.umd.edu), NSL, UMD
% v1.00: 01-Jun-97

  if std(x) < 1.0e-7
    x = x - mean(x);	% neutralization
  else 
    x = x - mean(x);	% neutralization
    x = x / std(x);	% normalization
  end;