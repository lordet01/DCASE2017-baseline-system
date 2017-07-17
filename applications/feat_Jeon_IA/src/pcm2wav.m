function pcm2wav(DIR_wavout, bit, p)

fout_read = fopen(DIR_wavout, 'rb');
if bit == 16
    out_full = fread(fout_read,inf, 'int16');
elseif bit == 24
    out_full = fread(fout_read,inf, 'bit24');
end
wavname = DIR_wavout;
out_full = out_full./(2^(bit-1)-1);
wavwrite(out_full, p.fs, bit, wavname);
fclose('all');

end