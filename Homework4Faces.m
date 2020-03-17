%% Start with a clean slate
clear all; close all; clc

%% cropped images
directories = dir('CroppedYale');
allFaces = [];

for i=1:length(directories)
    if directories(i).name == "." | directories(i).name == ".."
        continue
    end
    
    images = dir(strcat(directories(i).folder, '\', directories(i).name));
    for j=1:length(images)
        if images(j).name == "." | images(j).name == ".."
            continue
        end
        face = imread(strcat(images(j).folder, '\', images(j).name));
        allFaces = [allFaces reshape(face,192*168,1)];
    end
end

% normalize
original_Faces = allFaces;
data_size = size(allFaces);
allFaces = double(allFaces)./(data_size(2)-1);
mean_Faces = mean(allFaces);
allFaces = allFaces - mean_Faces;

[u,s,v] = svd(allFaces,'econ');

sig=diag(s);
figure(1)
subplot(1,2,1), plot(sig,'ko','Linewidth',[1.5])
subplot(1,2,2), semilogy(sig,'ko','Linewidth',[1.5])

figure(2)
subplot(1,2,1), plot(sig,'ko','Linewidth',[1.5])
set(gca,'Xlim',[0 100])
subplot(1,2,2), semilogy(sig,'ko','Linewidth',[1.5])
set(gca,'Xlim',[0 100])

energy = zeros(1,data_size(2));
for i=1:data_size(2)
    energy(i)=sum(sig(1:i))/sum(sig);
end
figure(3)
plot(energy)

figure(4)
num_modes = [1 5 100 1000];
for i=1:5
    for j=1:length(num_modes)
        subplot(5,5,i*5+j-5)
        approx1 = (u(:,1:num_modes(j))*s(1:num_modes(j),1:num_modes(j))*v(:,1:num_modes(j))' + mean_Faces).*(data_size(2)-1);
        approx1 = uint8(approx1);
        imshow(reshape(approx1(:,i*23),192,168))
    end
    subplot(5,5,i*5)
    imshow(reshape(original_Faces(:,i*23),192,168))
end

figure(5)
num_modes = [1 2 3 5 10 50 100 1000 2432];
for j=1:length(num_modes)
    subplot(3,3,j)
    % taken from code provided in class - otherwise images come out flipped
    ut1=reshape(u(:,num_modes(j)),192,168);
    ut2=ut1(192:-1:1,:);
    h = pcolor(ut2);
    colormap(hot)
    % remove edges, otherwise image details are hidden
    set(h, 'EdgeColor', 'none');
    title(num_modes(j))
end

%% uncropped images

images = dir('yalefaces');
allUFaces = [];

for j=1:length(images)
    if images(j).name == "." | images(j).name == ".."
        continue
    end
    face = imread(strcat(images(j).folder, '\', images(j).name));
    allUFaces = [allUFaces reshape(face,243*320,1)];
end

% normalize
original_UFaces = allUFaces;
Udata_size = size(allUFaces);
allUFaces = double(allUFaces)./(Udata_size(2)-1);
mean_UFaces = mean(allUFaces);
allUFaces = allUFaces - mean_UFaces;

[uu,us,uv] = svd(allUFaces,'econ');

usig=diag(us);
figure(6)
subplot(1,2,1), plot(usig,'ko','Linewidth',[1.5])
subplot(1,2,2), semilogy(usig,'ko','Linewidth',[1.5])

figure(7)
subplot(1,2,1), plot(usig,'ko','Linewidth',[1.5])
set(gca,'Xlim',[0 100])
subplot(1,2,2), semilogy(usig,'ko','Linewidth',[1.5])
set(gca,'Xlim',[0 100])

uenergy = zeros(1,Udata_size(2));
for i=1:Udata_size(2)
    uenergy(i)=sum(usig(1:i))/sum(usig);
end
figure(8)
plot(uenergy)

figure(9)
num_modes = [1 5 50 100];
for i=1:5
    for j=1:length(num_modes)
        subplot(5,5,i*5+j-5)
        approx1 = (uu(:,1:num_modes(j))*us(1:num_modes(j),1:num_modes(j))*uv(:,1:num_modes(j))' + mean_UFaces).*(Udata_size(2)-1);
        approx1 = uint8(approx1);
        imshow(reshape(approx1(:,i*6),243,320))
    end
    subplot(5,5,i*5)
    imshow(reshape(original_UFaces(:,i*6),243,320))
end

figure(10)
num_modes = [1 2 3 5 10 25 50 100 165];
for j=1:length(num_modes)
    subplot(3,3,j)
    % taken from code provided in class - otherwise images come out flipped
    ut1=reshape(uu(:,num_modes(j)),243,320);
    ut2=ut1(192:-1:1,:);
    h = pcolor(ut2);
    colormap(hot)
    % remove edges, otherwise image details are hidden
    set(h, 'EdgeColor', 'none');
    title(num_modes(j))
end