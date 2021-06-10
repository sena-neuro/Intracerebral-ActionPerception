clc;
clear variables;
close all;

diary findRejDataLog
filesep = '/';

currentDir = pwd;

% Input path
rejPath = fullfile(currentDir, '..', 'Data', 'ArtifactRejected', filesep);


% Output path
outPath = fullfile(currentDir, '..', 'Data', 'TF_Analyzed', filesep);

lowTbas = -500; % baseline period lower end (ms)
highTbas = 0; % baseline period higher end (ms)


% Get subject names
rejFolders = dir(rejPath);
subjNames = cell(40,1);
subjectNumber=0;

% Eliminate files with . and find subject names
for i=1:length(rejFolders)
    rejFolderNameChar = char(rejFolders(i).name);
    if(~strcmp(rejFolderNameChar(1),'.'))
        subjectNumber = subjectNumber + 1;
        subjNames{subjectNumber,1} = rejFolders(i).name;
    end
end

rejFlist=cell(subjectNumber,1);
rejFnames=cell(subjectNumber,1);
for i=1:subjectNumber
    
    fileList = dir([strcat(rejPath,subjNames{i}) filesep strcat(subjNames{i},'*.mat')]); % specify the subject .mat files
    for a = 1:length(fileList)
        rejFlist{a,1} = strcat(rejPath,subjNames{i}, filesep, fileList(a,1).name);
        rejFnames{a,1} = fileList(a,1).name;
    end
end

% Frequency analysis for each file in the filelist
for files=1:length(rejFlist)
   
    % load the artifact rejected data
    rejData = load(rejFlist{files}).D;

    %%%%%%%%%%%%%%%%%%%% Run time-frequency analysis on selected files %%%%%%%%%%%%%%%%%%%%
    ChanList=cell(length(rejData),1);
    for ch=1:length(rejData)
        if length(rejData{ch}.ChanType) == 4 && length(rejData{ch}.ChanName) < 5 % if the channel type is iEEG
            ChanList{ch,1} = rejData{ch,1}.ChanName;
            T{1,1}{ch} = rejData{ch,1}.ChanType;
        end
    end
    cond = '';

    us_idx = find(rejFnames{files} == '_'); % find the indices where there is '_' character
    cond = rejFnames{files}(us_idx(3)+1:us_idx(4)-1); % get the name of the condition
    
    fnmbase = rejFnames{files}(1:us_idx(1)-1); % get the name of the subjects
    outfnm = strcat(fnmbase,'_condition_',cond,'_seg_AR_tf.mat'); % combine the name of the subject with the condition above
    
    disp(strcat('Processing participant: ' ,fnmbase,' condition: ',cond)) 
    for i = 1:length(T{1,1})
        chans(i) = i;
    end
    
    for j = 1:length(chans) % run time-freq on all selected channels
        if isfield(rejData{chans(j),1},'Artifact_Removed_Wavelet_freq_flag')
            disp(strcat('Chan',chans(j),' ',' has already been processed by wavelet analysis'));
        else
            % Change this to semilog 50 steps from 5 to 152
            % Check all parameterss from Marije paper
            [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,fullg] = newtimef_pietro(...
                double(rejData{chans(j),1}.Segments2),...   % data in segments
                size(rejData{chans(j),1}.Segments2,1),...   % time points for segments
                rejData{chans(j),1}.TWin,...                % time range
                rejData{chans(j),1}.fs,...                  % sampling rate
                4,...                                 % number of cycles %'timesout',-50,...    % downsampling factor for wavelet panel NORMALMENTE -10!!! in BioPhy-20 %'scale','log',...     % log or linear per i dB
                'plotersp','off',...  % no plot for ersp panel
                'plotitc','off',...   % no plot for itc panel
                'nfreqs',50,...       % frequency resolution 23 [8-30] 30 [4-150]; 11 (50-150); 149 o 297 (4-300); 29 (10-150Hz); 11 (100Freq totali), 6 o 26 (50Freq totali), 91 (50-500); 47 (8-100); 72 (8-150)
                'freqs',[5 152],...  % frequency range 50-150Hz, 4-300, 50-150, 50-500 BioPhy:10-150Hz
                'freqscale','log',... %'linear' or log'
                'wletmethod','dftfilt3',...
                'rmerp','off',...% erp removed to work on the inter-trial coherence
                'baseline',[-500 0], ...  %);  
                'alpha',0.05,...      % significance threshold
                'mcorrect','fdr');    % multiple comparison correction ('fdr' or 'none')

            % Put the new fields
            rejData{chans(j),1}.Artifact_Removed_Wavelet_freq_flag = 1; % create a subfield in D that indicates that TF analysis was run
            rejData{chans(j),1}.AR_TWavbaseline=[lowTbas highTbas];
            rejData{chans(j),1}.AR_ersp = a1;
            rejData{chans(j),1}.AR_itc = a2;
            rejData{chans(j),1}.AR_powbase = a3;
            rejData{chans(j),1}.AR_times = a4;
            rejData{chans(j),1}.AR_freqs = a5;
            rejData{chans(j),1}.AR_erspboot = a6;
            rejData{chans(j),1}.AR_itcboot = a7;
            rejData{chans(j),1}.AR_tfX = a8;
            rejData{chans(j),1}.AR_maskersp = a9;
            rejData{chans(j),1}.AR_maskitc = a10;
            
            % Calculate power for this channel
            tfx = rejData{chans(j),1}.AR_tfX;
            if ~isempty(tfx)
                power = abs(tfx);
                
                % save the power infomation on new field in the struct
                rejData{chans(j),1}.AR_power = power;
            end
        end
    end
    % open folder for each subject if there is not
    if ~exist(strcat(outPath,filesep,fnmbase), 'dir')
        mkdir(strcat(outPath,filesep,fnmbase))
    end
    D = rejData;
    save(strcat(outPath,fnmbase,filesep,outfnm),'D')
    clear rejData ChanList T chans SignList SL_idx %check
    fclose('all');
end