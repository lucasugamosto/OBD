%Implementazione di una rete neurale
%Studente: Luca Sugamosto, matricola 0324613

clear
close all
clc

%Seed che è usato per generare dei numeri casuali, e quest'ultimi cambiano
%se e solo se viene cambiato il seed, cioè l'input della funzione 'rng'
rng(1);         

fprintf("Caricamento ed inizializzazione delle matrici per il training set e testing set\n")

%Caricamento del DATA SET, quindi del Training set e del Test set
load trainingSet_codrna.mat;
load testingSet_codrna.mat;

%Unione in un'unica grande matrice dei due Data Set caricati
finalDataSet = vertcat(DatasetsbinarycodrnaTraining,DatasetsbinarycodrnaTesting);

%Inizializzazione del numero di campioni e del numero di features del
%singolo campione
matrixSize = size(finalDataSet);
numSamples = matrixSize(1);
numFeatures = matrixSize(2) - 1;

%--------------------------------------------------------------------------

%Randomizzazione dei campioni e delle etichette associatevi per mezzo della
%funzione 'randsample', la quale genera dei numeri casuali non ripetuti
indexList = randsample(numSamples,numSamples);
dataSet = zeros(numSamples,matrixSize(2));
for i = 1:numSamples
    for j = 1:matrixSize(2)
        dataSet(i,j) = finalDataSet(indexList(i),j);
    end
end

%Calcolo della dimensione del TRAINING SET (circa il 72%)
trainingSetSamples = size(DatasetsbinarycodrnaTraining,1);
%Calcolo della dimensione del TESTING SET (circa il 28%)  
testingSetSamples = size(DatasetsbinarycodrnaTesting,1);

%Suddivisione del 'dataSet' in due elementi, la matrice dei campioni X e il
%vettore delle etichette Y
XDataSet = dataSet(:,2:matrixSize(2));
YDataSet = dataSet(:,1);

%Normalizzazione del DATA SET usando due diverse tipologie:
%1) Normalizzazione Max-Min;
%2) Standardizzazione (normalizzazione con media-varianza).

%1) Caso 1: NORMALIZZAZIONE MAX-MIN
%Normalizzazione delle componenti di ingresso tramite la funzione
%'max(X,[],2)' che calcola il massimo per ogni riga e 'min(X,[],2)' che
%calcola il minimo per ogni riga
% XDataSet = transpose(XDataSet);
% maxValue = max(XDataSet,[],2);
% minValue = min(XDataSet,[],2);
% XDataSet = (XDataSet - minValue)./(maxValue - minValue);

%2) Caso 2: STANDARDIZZAZIONE
meanValue = mean(XDataSet);
varianceValue = std(XDataSet);
XDataSet = (XDataSet - meanValue)./varianceValue;
XDataSet = transpose(XDataSet);

%Divisione del DATA SET in due sotto-insiemi: Training Set e Test Set
%                ----------TRAINING SET----------

%Ottengo un insieme con i campioni disposti in ordine sparso per aumentare
%la generalizzazione dei risultati
XTrain = XDataSet(:,1:trainingSetSamples);
YTrain = YDataSet(1:trainingSetSamples,1);

%                ----------TEST SET----------

%Creazione delle matrici per il test
XTest = XDataSet(:,trainingSetSamples+1:end);
YTest = YDataSet(trainingSetSamples+1:end,1);

%Calcolo del numero di etichette in uscita (cioè delle classi di uscita)
numLabels = length(unique(YTrain));

%--------------------------------------------------------------------------

fprintf('\nInizializzazione della rete neurale e fase di AGGIORNAMENTO DEI PESI\n');

%Inizializzazione della rete neurale e di tutti i suoi parametri
reteNeurale = createNeuralNetwork(numFeatures,trainingSetSamples,numLabels,YTrain);
reteNeurale = reteNeurale.weightInitialization();

%Aggiornamento delle matrici dei pesi W e dei vettori di bias B
reteNeurale = reteNeurale.weigthUpdate(XTrain,YTrain);

%Graficazione dell'andamento del valore di accuratezza e di perdita nel
%nello studio del Training Set
subplot(2,1,1);
plot(reteNeurale.accuracyVector,Color='r');
grid on
title('Accuracy Training Set');
xlabel('Epochs');
ylabel('Accuracy');
subplot(2,1,2);
plot(reteNeurale.lossVector,Color='b');
grid on
title('Loss Training Set');
xlabel('Epochs');
ylabel('Loss');

%--------------------------------------------------------------------------

fprintf("\nCalcolo del valore di ACCURACY sul Test Set\n")

%Esecuzione del forward propagation sul Test Set usando le matrici dei pesi
%W e i vettori di bias B calcolati in precedenza
reteNeurale = reteNeurale.accuracyTest(XTest,YTest);
