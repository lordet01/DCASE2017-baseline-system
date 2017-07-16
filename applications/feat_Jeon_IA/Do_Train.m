% /*********************************************
% 	Sound Event Detection & Separation Engine
%    - Basis Trainer (SNMF + MLD)
%
% 	Intelligent Acoustics
%    - Kwang Myung Jeon
% ***********************************************/


clear; clc;
%--------------------------------------------------------------------------
addpath('exp_cond');

%-----Choose one of settings-----
addpath('settings');
SED_initial_setting_SNMF;

dir_DB_TRAIN = '../data/TUT-rare-sound-events-2017-development/data/source_data';

%% Train event bases
EVENT_NAME = {'babycry', 'glassbreak', 'gunshot'};
EVENT_NUM = size(EVENT_NAME,2);
dir_Basis = 'basis'; mkdir(dir_Basis);

%Train multiple event bases then merge into one
DB_path_list = cell(1, EVENT_NUM);
B_class_event_full = cell(1, EVENT_NUM);
for e=1:EVENT_NUM
    DB_path_list{e} = [dir_DB_TRAIN,'/events/',EVENT_NAME{e}];
    B_class_event_full{e} = ['event_',EVENT_NAME{e}]; mkdir([dir_Basis,'/',B_class_event_full{e}]);
end

DC_freq_set = [80 80 80];
VAD_set = [0 0 0];
[B_x_DFT, B_x_Mel, ~, ~, p] = run_basis_train(DB_path_list, DC_freq_set, ...
                                           VAD_set, EVENT_NUM, B_class_event_full, p.R_x, p);
   
%% Train background noise bases
DB_path_list = cell(1,1);
B_class_event_full = cell(1,1);
DB_path_list{1} = [dir_DB_TRAIN,'/bgs/audio'];
B_class_event_full{1} = 'bgn_DCASE2017'; mkdir([dir_Basis,'/',B_class_event_full{1}]);
DC_freq_set = [80];
VAD_set = [0];
[B_d_DFT, B_d_Mel, ~, ~] = run_basis_train(DB_path_list, DC_freq_set, ...
                                           VAD_set, 1, B_class_event_full, p.R_d, p);
                                
