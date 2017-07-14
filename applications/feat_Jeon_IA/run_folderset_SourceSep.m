
tic;
in_path = './wav_sample';
out_path = './wav_sample/proc_NAT_MLD';
mkdir(out_path);

[file_list]=dir([in_path,'/*.wav']);
for i=1:length(file_list)
    filewise_SourceSep([in_path,'/',file_list(i).name], [out_path,'/',file_list(i).name])
end
toc;