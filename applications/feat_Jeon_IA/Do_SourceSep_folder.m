% /*********************************************
% 	Sound Event Detection & Separation Engine
%    - Folder-wise Source Seperator (Audio to Audio)
%
% 	Intelligent Acoustics
%    - Kwang Myung Jeon
% ***********************************************/


tic;
in_path = './wav_sample';
out_path = './wav_sample/proc_MLD_ODL_up1_target_event_lowB_Mel_SC_multinoiseB';
mkdir(out_path);


EVENT_LIST = {'babycry', 'glassbreak', 'gunshot'};
[file_list]=dir([in_path,'/*.wav']);
for i=1:length(file_list)
    K = strfind(file_list(i).name, 'babycry');
    if K >= 1
        filewise_SourceSep([in_path,'/',file_list(i).name], [out_path,'/',file_list(i).name], EVENT_LIST, 'babycry');
    end
    
    K = strfind(file_list(i).name, 'glassbreak');
    if K >= 1
        filewise_SourceSep([in_path,'/',file_list(i).name], [out_path,'/',file_list(i).name], EVENT_LIST, 'glassbreak');
    end
    
    K = strfind(file_list(i).name, 'gunshot');
    if K >= 1
        filewise_SourceSep([in_path,'/',file_list(i).name], [out_path,'/',file_list(i).name], EVENT_LIST, 'gunshot');
    end
end
toc;