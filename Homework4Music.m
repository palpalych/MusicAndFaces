%% Start with a clean slate
clear all; close all; clc

%% Band Classification
songsFolder = dir('Music\Ambient');
ambientSongs = [];

for i=1:length(songsFolder)
    if songsFolder(i).name == "." | songsFolder(i).name == ".."
        continue
    end
    % I'm assuming that the left vs right stereo aren't noticeably important for
    % these clips. Additionally reducing the signal frequency for a more
    % manageable amount of data
    [m,Fs] = audioread(strcat(songsFolder(i).folder, '\', songsFolder(i).name));
    ambientSongs = [ambientSongs reshape(spectrogram(m(1:2:Fs*5+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*15:2:Fs*20+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*30:2:Fs*35+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*55:2:Fs*60+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*65:2:Fs*70+1,1)),16385*8,1)];
end

songsFolder = dir('Music\Blues');
bluesSongs = [];

for i=1:length(songsFolder)
    if songsFolder(i).name == "." | songsFolder(i).name == ".."
        continue
    end
    [m,Fs] = audioread(strcat(songsFolder(i).folder, '\', songsFolder(i).name));
    bluesSongs = [bluesSongs reshape(spectrogram(m(1:2:Fs*5+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*15:2:Fs*20+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*30:2:Fs*35+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*55:2:Fs*60+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*65:2:Fs*70+1,1)),16385*8,1)];
end

songsFolder = dir('Music\Rock');
rockSongs = [];

for i=1:length(songsFolder)
    if songsFolder(i).name == "." | songsFolder(i).name == ".."
        continue
    end
    [m,Fs] = audioread(strcat(songsFolder(i).folder, '\', songsFolder(i).name));
    rockSongs = [rockSongs reshape(spectrogram(m(1:2:Fs*5+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*15:2:Fs*20+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*30:2:Fs*35+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*55:2:Fs*60+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*65:2:Fs*70+1,1)),16385*8,1)];
end

allSongs = abs([ambientSongs bluesSongs rockSongs]);
mean_allSongs = mean(allSongs);
allSongsCentered = allSongs - mean_allSongs;
[u,s,v] = svd(allSongsCentered,'econ');

sig=diag(s);
figure(1)
subplot(1,2,1), plot(sig,'ko','Linewidth',[1.5])
subplot(1,2,2), semilogy(sig,'ko','Linewidth',[1.5])
%%
startmode = 1;
endmodes = 80;
Transform = s(startmode:endmodes,startmode:endmodes)\u(:,startmode:endmodes)';

random_ambient = randperm(40);
random_blues = randperm(45);
random_rock = randperm(45);

ambient = v(1:40,startmode:endmodes);
blues = v(41:85,startmode:endmodes);
rock = v(86:end,startmode:endmodes);

xtrain=[ambient(random_ambient(1:35),:); blues(random_blues(1:40),:); rock(random_rock(1:40),:)];
xtest=[ambient(random_ambient(36:end),:); blues(random_blues(41:end),:); rock(random_rock(41:end),:)];
xexpected=string(zeros(15,1));
xexpected(1:5)="ambient";
xexpected(6:10)="blues";
xexpected(11:15)="rock";
ctrain=string(zeros(115,1));
ctrain(1:35)="ambient";
ctrain(36:75)="blues";
ctrain(76:end)="rock";

% try both a Gaussian Naive Bayes model and an LDA
nb=fitcnb(xtrain,ctrain);
trainAccuracyBayes = sum(nb.predict(xtest)==xexpected)/15
trainAccuracyLDA = sum(classify(xtest,xtrain,ctrain)==xexpected)/15

correctBayes = 0;
correctLda = 0;
% take a completely unused Ambient song and test against it
[m,Fs] = audioread('.\Music\Test\Ambient.mp3');
testAmbient = [reshape(spectrogram(m(1:2:Fs*5+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*15:2:Fs*20+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*30:2:Fs*35+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*55:2:Fs*60+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*65:2:Fs*70+1,1)),16385*8,1)];
testAmbient = abs(testAmbient);
testAmbient = testAmbient - mean(testAmbient);
transformedAmbient = (Transform * testAmbient)';
correctBayes = correctBayes + sum(nb.predict(transformedAmbient)=="ambient");
correctLda = correctLda + sum(classify(transformedAmbient,xtrain,ctrain)=="ambient");

% Repeat for a Rock song
[m,Fs] = audioread('.\Music\Test\Rock.mp3');
testRock = [reshape(spectrogram(m(1:2:Fs*5+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*15:2:Fs*20+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*30:2:Fs*35+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*55:2:Fs*60+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*65:2:Fs*70+1,1)),16385*8,1)];
testRock = abs(testRock);
testRock = testRock - mean(testRock);
transformedRock = (Transform * testRock)';
correctBayes = correctBayes + sum(nb.predict(transformedRock)=="rock");
correctLda = correctLda + sum(classify(transformedRock,xtrain,ctrain)=="rock");

% Repeat for a Blues song
[m,Fs] = audioread('.\Music\Test\Blues.mp3');
testBlues = [reshape(spectrogram(m(1:2:Fs*5+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*15:2:Fs*20+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*30:2:Fs*35+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*55:2:Fs*60+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*65:2:Fs*70+1,1)),16385*8,1)];
testBlues = abs(testBlues);
testBlues = testBlues - mean(testBlues);
transformedBlues = (Transform * testBlues)';
correctBayes = correctBayes + sum(nb.predict(transformedBlues)=="blues");
correctLda = correctLda + sum(classify(transformedBlues,xtrain,ctrain)=="blues");

BayesAccuracy = correctBayes / 15
LDAAccuracy = correctLda / 15

%% Classification within genre

songsFolder = dir('Music\Rock2');
rock2Songs = [];

for i=1:length(songsFolder)
    if songsFolder(i).name == "." | songsFolder(i).name == ".."
        continue
    end
    [m,Fs] = audioread(strcat(songsFolder(i).folder, '\', songsFolder(i).name));
    rock2Songs = [rock2Songs reshape(spectrogram(m(1:2:Fs*5+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*15:2:Fs*20+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*30:2:Fs*35+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*55:2:Fs*60+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*65:2:Fs*70+1,1)),16385*8,1)];
end

songsFolder = dir('Music\Rock3');
rock3Songs = [];

for i=1:length(songsFolder)
    if songsFolder(i).name == "." | songsFolder(i).name == ".."
        continue
    end
    [m,Fs] = audioread(strcat(songsFolder(i).folder, '\', songsFolder(i).name));
    rock3Songs = [rock3Songs reshape(spectrogram(m(1:2:Fs*5+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*15:2:Fs*20+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*30:2:Fs*35+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*55:2:Fs*60+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*65:2:Fs*70+1,1)),16385*8,1)];
end

allSongs = abs([rock3Songs rock2Songs rockSongs]);
mean_allSongs = mean(allSongs);
allSongsCentered = allSongs - mean_allSongs;
[ru,rs,rv] = svd(allSongsCentered,'econ');

sig=diag(rs);
figure(2)
subplot(1,2,1), plot(sig,'ko','Linewidth',[1.5])
subplot(1,2,2), semilogy(sig,'ko','Linewidth',[1.5])
%%
startmode = 1;
endmodes = 8;
Transform = rs(startmode:endmodes,startmode:endmodes)\ru(:,startmode:endmodes)';

random_rock3 = randperm(45);
random_rock2 = randperm(45);
random_rock = randperm(45);

rock3 = rv(1:45,startmode:endmodes);
rock2 = rv(46:90,startmode:endmodes);
rock = rv(91:end,startmode:endmodes);

xtrain=[rock3(random_rock3(1:40),:); rock2(random_rock2(1:40),:); rock(random_rock(1:40),:)];
xtest=[rock3(random_rock3(41:end),:); rock2(random_rock2(41:end),:); rock(random_rock(41:end),:)];
xexpected=string(zeros(15,1));
xexpected(1:5)="rock3";
xexpected(6:10)="rock2";
xexpected(11:15)="rock";
ctrain=string(zeros(120,1));
ctrain(1:40)="rock3";
ctrain(41:80)="rock2";
ctrain(81:end)="rock";

% try both a Gaussian Naive Bayes model and an LDA
nb=fitcnb(xtrain,ctrain);
trainAccuracyBayes = sum(nb.predict(xtest)==xexpected)/15
trainAccuracyLDA = sum(classify(xtest,xtrain,ctrain)==xexpected)/15

correctBayes = 0;
correctLda = 0;
% take a completely unused Rock song and test against it
[m,Fs] = audioread('.\Music\Test\Rock.mp3');
testRock = [reshape(spectrogram(m(1:2:Fs*5+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*15:2:Fs*20+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*30:2:Fs*35+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*55:2:Fs*60+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*65:2:Fs*70+1,1)),16385*8,1)];
testRock = abs(testRock);
testRock = testRock - mean(testRock);
transformedRock = (Transform * testRock)';
correctBayes = correctBayes + sum(nb.predict(transformedRock)=="rock");
correctLda = correctLda + sum(classify(transformedRock,xtrain,ctrain)=="rock");

% Repeat for an unused Rock2 song
[m,Fs] = audioread('.\Music\Test\Rock2.mp3');
testRock = [reshape(spectrogram(m(1:2:Fs*5+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*15:2:Fs*20+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*30:2:Fs*35+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*55:2:Fs*60+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*65:2:Fs*70+1,1)),16385*8,1)];
testRock = abs(testRock);
testRock = testRock - mean(testRock);
transformedRock = (Transform * testRock)';
correctBayes = correctBayes + sum(nb.predict(transformedRock)=="rock2");
correctLda = correctLda + sum(classify(transformedRock,xtrain,ctrain)=="rock2");

% Repeat for an unused Rock3 song
[m,Fs] = audioread('.\Music\Test\Rock3.mp3');
testRock = [reshape(spectrogram(m(1:2:Fs*5+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*15:2:Fs*20+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*30:2:Fs*35+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*55:2:Fs*60+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*65:2:Fs*70+1,1)),16385*8,1)];
testRock = abs(testRock);
testRock = testRock - mean(testRock);
transformedRock = (Transform * testRock)';
correctBayes = correctBayes + sum(nb.predict(transformedRock)=="rock3");
correctLda = correctLda + sum(classify(transformedRock,xtrain,ctrain)=="rock3");

BayesAccuracy = correctBayes / 15
LDAAccuracy = correctLda / 15

%% Genre Classification

songsFolder = dir('Music\Jazz');
jazzSongs = [];

for i=1:length(songsFolder)
    if songsFolder(i).name == "." | songsFolder(i).name == ".."
        continue
    end
    [m,Fs] = audioread(strcat(songsFolder(i).folder, '\', songsFolder(i).name));
    jazzSongs = [jazzSongs reshape(spectrogram(m(1:2:Fs*5+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*15:2:Fs*20+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*30:2:Fs*35+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*55:2:Fs*60+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*65:2:Fs*70+1,1)),16385*8,1)];
end

songsFolder = dir('Music\Jazz2');
jazz2Songs = [];

for i=1:length(songsFolder)
    if songsFolder(i).name == "." | songsFolder(i).name == ".."
        continue
    end
    [m,Fs] = audioread(strcat(songsFolder(i).folder, '\', songsFolder(i).name));
    jazz2Songs = [jazz2Songs reshape(spectrogram(m(1:2:Fs*5+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*15:2:Fs*20+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*30:2:Fs*35+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*55:2:Fs*60+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*65:2:Fs*70+1,1)),16385*8,1)];
end

songsFolder = dir('Music\Jazz3');
jazz3Songs = [];

for i=1:length(songsFolder)
    if songsFolder(i).name == "." | songsFolder(i).name == ".."
        continue
    end
    [m,Fs] = audioread(strcat(songsFolder(i).folder, '\', songsFolder(i).name));
    jazz3Songs = [jazz3Songs reshape(spectrogram(m(1:2:Fs*5+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*15:2:Fs*20+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*30:2:Fs*35+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*55:2:Fs*60+1,1)),16385*8,1)];
end

songsFolder = dir('Music\Ambient2');
ambient2Songs = [];

for i=1:length(songsFolder)
    if songsFolder(i).name == "." | songsFolder(i).name == ".."
        continue
    end
    [m,Fs] = audioread(strcat(songsFolder(i).folder, '\', songsFolder(i).name));
    ambient2Songs = [ambient2Songs reshape(spectrogram(m(1:2:Fs*5+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*15:2:Fs*20+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*30:2:Fs*35+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*55:2:Fs*60+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*65:2:Fs*70+1,1)),16385*8,1)];
end

songsFolder = dir('Music\Ambient3');
ambient3Songs = [];

for i=1:length(songsFolder)
    if songsFolder(i).name == "." | songsFolder(i).name == ".."
        continue
    end
    [m,Fs] = audioread(strcat(songsFolder(i).folder, '\', songsFolder(i).name));
    ambient3Songs = [ambient3Songs reshape(spectrogram(m(1:2:Fs*5+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*15:2:Fs*20+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*30:2:Fs*35+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*55:2:Fs*60+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*65:2:Fs*70+1,1)),16385*8,1)];
end

allSongs = abs([jazzSongs jazz2Songs jazz3Songs rock3Songs rock2Songs rockSongs ambientSongs ambient2Songs ambient3Songs]);
mean_allSongs = mean(allSongs);
allSongsCentered = allSongs - mean_allSongs;
[ru,rs,rv] = svd(allSongsCentered,'econ');

sig=diag(rs);
figure(3)
subplot(1,2,1), plot(sig,'ko','Linewidth',[1.5])
subplot(1,2,2), semilogy(sig,'ko','Linewidth',[1.5])
%%
startmode = 1;
endmodes = 250;
Transform = rs(startmode:endmodes,startmode:endmodes)\ru(:,startmode:endmodes)';

random_jazz = randperm(140);
random_rock = randperm(135);
random_ambient = randperm(140);

jazz = rv(1:140,startmode:endmodes);
rock = rv(141:275,startmode:endmodes);
ambient = rv(276:end,startmode:endmodes);

xtrain=[jazz(random_jazz(1:120),:); rock(random_rock(1:120),:); ambient(random_ambient(1:120),:)];
xtest=[jazz(random_jazz(121:end),:); rock(random_rock(121:end),:); ambient(random_ambient(121:end),:)];
xexpected=string(zeros(55,1));
xexpected(1:20)="jazz";
xexpected(21:35)="rock";
xexpected(36:55)="ambient";
ctrain=string(zeros(360,1));
ctrain(1:120)="jazz";
ctrain(121:240)="rock";
ctrain(241:end)="ambient";

% try both a Gaussian Naive Bayes model and an LDA
nb=fitcnb(xtrain,ctrain);
trainAccuracyBayes = sum(nb.predict(xtest)==xexpected)/55
trainAccuracyLDA = sum(classify(xtest,xtrain,ctrain)==xexpected)/55

correctBayes = 0;
correctLda = 0;
% take a completely unused Rock song and test against it
[m,Fs] = audioread('.\Music\Test\Rock.mp3');
testRock = [reshape(spectrogram(m(1:2:Fs*5+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*15:2:Fs*20+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*30:2:Fs*35+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*55:2:Fs*60+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*65:2:Fs*70+1,1)),16385*8,1)];
testRock = abs(testRock);
testRock = testRock - mean(testRock);
transformedRock = (Transform * testRock)';
correctBayes = correctBayes + sum(nb.predict(transformedRock)=="rock");
correctLda = correctLda + sum(classify(transformedRock,xtrain,ctrain)=="rock");

% Repeat for an unused Rock2 song
[m,Fs] = audioread('.\Music\Test\Rock2.mp3');
testRock = [reshape(spectrogram(m(1:2:Fs*5+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*15:2:Fs*20+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*30:2:Fs*35+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*55:2:Fs*60+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*65:2:Fs*70+1,1)),16385*8,1)];
testRock = abs(testRock);
testRock = testRock - mean(testRock);
transformedRock = (Transform * testRock)';
correctBayes = correctBayes + sum(nb.predict(transformedRock)=="rock2");
correctLda = correctLda + sum(classify(transformedRock,xtrain,ctrain)=="rock2");

% Repeat for an unused Rock3 song
[m,Fs] = audioread('.\Music\Test\Rock3.mp3');
testRock = [reshape(spectrogram(m(1:2:Fs*5+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*15:2:Fs*20+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*30:2:Fs*35+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*55:2:Fs*60+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*65:2:Fs*70+1,1)),16385*8,1)];
testRock = abs(testRock);
testRock = testRock - mean(testRock);
transformedRock = (Transform * testRock)';
correctBayes = correctBayes + sum(nb.predict(transformedRock)=="rock3");
correctLda = correctLda + sum(classify(transformedRock,xtrain,ctrain)=="rock3");

% Repeat for an unused Jazz song
[m,Fs] = audioread('.\Music\Test\Jazz.mp3');
testJazz = [reshape(spectrogram(m(1:2:Fs*5+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*15:2:Fs*20+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*30:2:Fs*35+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*55:2:Fs*60+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*65:2:Fs*70+1,1)),16385*8,1)];
testJazz = abs(testJazz);
testJazz = testJazz - mean(testJazz);
transformedJazz = (Transform * testJazz)';
correctBayes = correctBayes + sum(nb.predict(transformedJazz)=="jazz");
correctLda = correctLda + sum(classify(transformedJazz,xtrain,ctrain)=="jazz");

% Repeat for an unused Ambient song
[m,Fs] = audioread('.\Music\Test\Ambient.mp3');
testAmbient = [reshape(spectrogram(m(1:2:Fs*5+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*15:2:Fs*20+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*30:2:Fs*35+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*55:2:Fs*60+1,1)),16385*8,1) ...
        reshape(spectrogram(m(Fs*65:2:Fs*70+1,1)),16385*8,1)];
testAmbient = abs(testAmbient);
testAmbient = testAmbient - mean(testAmbient);
transformedAmbient = (Transform * testAmbient)';
correctBayes = correctBayes + sum(nb.predict(transformedAmbient)=="ambient");
correctLda = correctLda + sum(classify(transformedAmbient,xtrain,ctrain)=="ambient");

BayesAccuracy = correctBayes / 25
LDAAccuracy = correctLda / 25