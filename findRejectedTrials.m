clc;
clear;

%addpath('/auto/k2/oelmas/eeglab14_1_2b')
diary findRejDataLog

%pth='/auto/data2/oelmas/EEG_AgentPerception_NAIVE/Data/';
rej_pth='/Users/huseyinelmas/CCNLAB/IntacerebralEEG_ActionBase/ArtifactRejected/';
nonrej_pth='/Users/huseyinelmas/CCNLAB/IntacerebralEEG_ActionBase/NoArtRejection/';


% We need a map that has structs for keys and values, keys will hold 
% Which file the struct (or cell array) belongs to and struct(or cell array) will hold 163 structs 
% each has indices of the non-rejected trials
% If no trial rejection made there should be a flag that can be easily
% accessed


% Get subject names
rej_folders = dir(rej_pth);
nonrej_folders = dir(nonrej_pth);

subj_names = {};
subject_number=0;
% Eliminate files with . 
for i=1:length(rej_folders)
    if(~startsWith(rej_folders(i).name,'.'))
        subject_number = subject_number+1;
        subj_names{subject_number} = rej_folders(i).name;
    end
end


% Get non-rejected file names
RejFlist={};
for i=1:subject_number
    FileList = dir([strcat(rej_pth,subj_names{i}) filesep strcat(subj_names{i},'*.mat')]); % specify the subject .mat files
    for a = 1:length(FileList)
        RejFlist{a,1} = strcat(rej_pth,subj_names{i},'/',FileList(a,1).name);
    end
end

% Get non-rejected file names
NonRejFlist={};
NonRejFnames={};
for i=1:subject_number
    FileList = dir([strcat(nonrej_pth,subj_names{i}) filesep strcat(subj_names{i},'*.mat')]); % specify the subject .mat files
    for a = 1:length(FileList)
        NonRejFlist{a,1} = strcat(nonrej_pth,subj_names{i},'/',FileList(a,1).name);
        NonRejFnames{a,1} = FileList(a,1).name;
    end
end

% We know the number of values and keys so fix it here
valueCellArray = cell(length(NonRejFlist),1);
keyCellArray = cell(length(NonRejFlist),1);

for i=1:length(RejFlist)
    % load the artifact rejected data
    rejData = load(RejFlist{i}).D;
    
    % Cell array to hold rejected trials list for each data file
    keepCellArray = cell(162,1);
    for j=1:length(rejData)
        if(strcmp(rejData{j}.ChanType,'iEEG'))
            % No trials were rejected, create a struct with only one field
            if(~isfield(rejData{j}, 'RemovedTrials') || rejData{j}.RemovedTrials == 0)
                keepCellArray{j} = 0;
            else
                % Load Non Artifact recejted data
                nonRejData = load(NonRejFlist{i}).D;

                % Compare trials and create a struct to put it inside the cell
                % array
                keep = keepTrials(rejData{j}.Segments2,nonRejData{j}.Segments2);
                keepCellArray{j} = keep;
            end
        end
    end
    valueCellArray{i} = keepCellArray;
    keyCellArray{i} = NonRejFlist{i};
end

map = containers.Map(NonRejFnames,valueCellArray);
save('map')