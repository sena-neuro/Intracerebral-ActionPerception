clc;
clear variables;
close all;

diary frequency_analysis_log
filesep = '/';

all_paths = genpath('/auto/k2/oelmas/eeglab2019_1-2');
addpath(all_paths);


% currentDir = pwd;

% Input path
direc = '/auto/data2/oelmas/Intracerebral';
rejPath = fullfile(direc, 'Data', 'ArtifactRejected', filesep);
% rejPath = fullfile(currentDir, '..', 'Data', 'ArtifactRejected', filesep);


% Output path
outPath = fullfile(direc, 'Data', 'TF_Analyzed', filesep);
% outPath = fullfile(currentDir, '..', 'Data', 'TF_Analyzed', filesep);

lowTbas = -500; % baseline period lower end (ms)
highTbas = 0; % baseline period higher end (ms)


% Get subject names
rejFolders = dir(rejPath);
subjNames = {};
subjectNumber=0;

% Eliminate files with . and find subject names
for i=1:length(rejFolders)
    rejFolderNameChar = char(rejFolders(i).name);
    if(~strcmp(rejFolderNameChar(1),'.'))
        subjectNumber = subjectNumber + 1;
        subjNames{subjectNumber,1} = rejFolders(i).name;
    end
end

rejFlist={};
rejFnames={};
ind=1;
for i=1:subjectNumber

    fileList = dir([strcat(rejPath,subjNames{i}) filesep strcat('*.mat')]); % specify the subject .mat files
    for a = 1:length(fileList)
        rejFlist{ind,1} = strcat(rejPath,subjNames{i}, filesep, fileList(a,1).name);
        rejFnames{ind,1} = fileList(a,1).name;
        ind=ind+1;
    end
end
% Frequency analysis for each file in the filelist
for files=1:length(rejFlist)

    % load the artifact rejected data
    ar = load(rejFlist{files});
    rejData = ar.D;

    %%%%%%%%%%%%%%%%%%%% Run time-frequency analysis on selected files %%%%%%%%%%%%%%%%%%%%
    ChanList=cell(length(rejData),1);
    eegChanIdx = 1;
    cond = '';

    str = rejFnames{files};
    expression = '_condition_\d\d\d';
    [startIndex, endIndex] = regexp(str, expression);
    cond = str(endIndex-2:endIndex); % get the name of the condition

    us_idx = find(rejFnames{files} == '_'); % find the indices where there is '_' character
    fnmbase = rejFnames{files}(1:us_idx(1)-1); % get the name of the subjects
    outfnm = strcat(fnmbase,'_condition_',cond,'_seg_AR_tf.mat'); % combine the name of the subject with the condition above

    if (exist(strcat(outPath,fnmbase,filesep,outfnm)))
        continue;
    end

    disp(strcat('Processing participant: ' ,fnmbase,' condition: ',cond))

    for j = 1:length(rejData) % run time-freq on all selected channels
        if ~(strcmp(rejData{j}.ChanType, 'iEEG') && length(rejData{j}.ChanName) < 5) % if the channel type is iEEG
            continue;
        end
        if isfield(rejData{j,1},'AR_tfX')
            disp(strcat('Chan',j,' ',' has already been processed by wavelet analysis'));
        else
            % Change this to semilog 50 steps from 5 to 152
            % Check all parameterss from Marije paper
            [ersp,itc,powbase,times,freqs,erspboot,itcboot,tfdata] = newtimef(...
            double(rejData{j,1}.Segments2),...   % data in segments
                size(rejData{j,1}.Segments2,1),...   % time points for segments
                rejData{j,1}.TWin,...                % time range
                rejData{j,1}.fs,...                  % sampling rate
                4,...                                 % number of cycles %'timesout',-50,...    % downsampling factor for wavelet panel NORMALMENTE -10!!! in BioPhy-20 %'scale','log',...     % log or linear per i dB
                'plotersp','off',...  % no plot for ersp panel
                'plotitc','off',...   % no plot for itc panel
                'nfreqs',50,...       % frequency resolution 23 [8-30] 30 [4-150]; 11 (50-150); 149 o 297 (4-300); 29 (10-150Hz); 11 (100Freq totali), 6 o 26 (50Freq totali), 91 (50-500); 47 (8-100); 72 (8-150)
                'freqs',[5 152],...  % frequency range 50-150Hz, 4-300, 50-150, 50-500 BioPhy:10-150Hz
                'freqscale','log',... %'linear' or log'
                'wletmethod','dftfilt3',...
                'rmerp','off',...% erp removed to work on the inter-trial coherence
                'baseline',[-500 0],...
                 'verbose','off');

            % Put the new fields
            rejData{j,1}.Artifact_Removed_Wavelet_freq_flag = 1; % create a subfield in D that indicates that TF analysis was run
            rejData{j,1}.AR_TWavbaseline=[lowTbas highTbas];
            rejData{j,1}.AR_ersp = ersp;
            rejData{j,1}.AR_itc = itc;
            rejData{j,1}.AR_powbase = powbase;
            rejData{j,1}.AR_times = times;
            rejData{j,1}.AR_freqs = freqs;
            rejData{j,1}.AR_erspboot = erspboot;
            rejData{j,1}.AR_itcboot = itcboot;
            rejData{j,1}.AR_tfX = tfdata;

            % Calculate power for this channel
            tfx = rejData{j,1}.AR_tfX;
            if ~isempty(tfx)
             % save the power infomation on new field in the struct
                rejData{j,1}.AR_power = abs(tfx);

            end
        end
    end
    % open folder for each subject if there is not
    if ~exist(strcat(outPath,filesep,fnmbase), 'dir')
        mkdir(strcat(outPath,filesep,fnmbase))
    end
    D = rejData;
    save(strcat(outPath,fnmbase,filesep,outfnm),'D')
    clear ar rejData ChanList chans SignList SL_idx %check
end

