classdef createNeuralNetwork
    properties
        features                        %Numero di componenti del campione
        samples                         %Numero di campioni del dataset
        labels                          %Numero di etichette di uscita
        firstHiddenLayer                %Numero di neuroni per il primo strato nascosto
        secondHiddenLayer               %Numero di neuroni per il secondo strato nascosto
        numHiddenLayer                  %Numero di strati nascosti
        epochs                          %Numero di epoche totali
        n                               %Vettore contenente il numero di neuroni per ogni strato
        m                               %Numero di campioni per iterazione
        labelVector                     %Vettore delle etichette di uscita
        outputArray                     %Matrice contenente le uscite della propagazione in avanti
        
        W1                              %Matrice di dimensione OUTxIN
        W2                              %Matrice di dimensione OUTxIN
        W3                              %Matrice di dimensione OUTxIN
        B1                              %Matrice di dimensione OUTx1
        B2                              %Matrice di dimensione OUTx1
        B3                              %Matrice di dimensione OUTx1

        a1                              %Matrice di dimensione OUTxCAMPIONICONSIDERATI
        a2                              %Matrice di dimensione OUTxCAMPIONICONSIDERATI
        a3                              %Matrice di dimensione OUTxCAMPIONICONSIDERATI
        z1                              %Matrice di dimensione OUTxCAMPIONICONSIDERATI
        z2                              %Matrice di dimensione OUTxCAMPIONICONSIDERATI
        z3                              %Matrice di dimensione OUTxCAMPIONICONSIDERATI

        alpha                           %Passo di aggiornamento costante (Learning Rate)
        dss                             %Passo di aggiornamento diminishing step-size
        lossVector                      %Vettore contenente la perdita ad ogni iterazione
        accuracyVector                  %Vettore contenente l'accuratezza ad ogni iterazione
    end

    methods
        function obj = createNeuralNetwork(variable1,variable2,variable3,variable4)
            %Inizializzazione dei parametri della rete neurale e dei
            %vettori, usati per i calcoli ed il salvataggio dei dati nelle
            %successive funzioni
            obj.features = variable1;
            obj.samples = variable2;
            obj.labels = variable3;
            obj.labelVector = unique(variable4);            %Vettore contenente tutte le etichette di uscita ripetute una sola volta

            obj.firstHiddenLayer = 10;
            obj.secondHiddenLayer = 6;
            obj.numHiddenLayer = 2;
            obj.n = [obj.features, obj.firstHiddenLayer, obj.secondHiddenLayer, obj.labels];
            obj.m = 16;                                      %Caso del metodo STOCASTICO, in quanto si utilizzano solo alcuni campioni ad ogni iterazione
            % obj.m = obj.samples;                             %Caso del metodo BATCH, in quanto si utilizzano tutti i campioni ad ogni iterazione 
            obj.alpha = 0.005;                               %Inizializzazione del constant step-size
            % obj.dss = 0;                                    %Inizializzazione del diminishing step-size
            obj.epochs = 25;

            %Inizializzazione dei vettori usati per salvare i risultati
            obj.outputArray = [];
            obj.accuracyVector = [];
            obj.lossVector = [];
        end

        function obj = weightInitialization(obj)
            %Inizializzazione delle matrici dei pesi e dei vettori di bias
            %da aggiornare nella fase di training
            obj.W1 = randn(obj.n(2),obj.n(1));
            obj.W2 = randn(obj.n(3),obj.n(2));
            obj.W3 = randn(obj.n(4),obj.n(3));
            obj.B1 = zeros(obj.n(2),1);
            obj.B2 = zeros(obj.n(3),1);
            obj.B3 = zeros(obj.n(4),1);
        end

        function reluValue = reluFunction(~,z)
            %Funzione che riceve in ingresso una costante e restituisce
            %il massimo tra il valore di ingresso stesso e 0
            reluValue = max(0,z);
        end

        function gradientValue = gradientFunction(~,z)
            %Funzione che riceve in ingresso una costante e restituisce 1
            %se il valore di ingresso è positivo, 0 altrimenti
            gradientValue = double(z > 0);
        end

        function dEdy = derivativeComputation(obj,YTrue,j,sample)
            %Funzione che calcola la derivata della funzione di perdita
            %rispetto alle etichette di uscita
            sum1 = 0;
            for i = 1:obj.labels
                if (i ~= j)
                    sum1 = sum1 + YTrue(i,sample);
                end
            end
            denominatorSum = 0;
            for i = 1:obj.labels
                denominatorSum = denominatorSum + exp(obj.a3(i,sample));
            end

            sum2 = 0;
            for i = 1:obj.labels
                if (i ~= j)
                    sum2 = sum2 + exp(obj.a3(i,sample));
                end
            end
            %Derivata della funzione di perdita rispetto alle etichette j
            dEdy = (-(YTrue(j,sample)./denominatorSum)).*sum2;
            dEdy = dEdy + ((exp(obj.a3(j,sample)))./denominatorSum).*sum1;
        end

        function obj = errorFunction(obj,y)
            %Funzione che calcola l'errore di perdita per mezzo della
            %funzione di "cross-entropy"

            %Creazione del vettore delle etichette vere andando a creare
            %una matrice, in cui data una certa etichetta vera allora si
            %assegna valore 1 alla posizione associata a quella etichetta e
            %0 a tutte le altre
            YTrue = zeros(obj.labels,obj.samples);
            for i = 1:obj.samples
                index = find(obj.labelVector == y(i));
                YTrue(index,i) = 1;
            end

            %Calcolo del valore di perdita associato a tutti i campioni del
            %data set
            sum1 = 0;
            for ii = 1:obj.samples
                sum2 = 0;
                for hh = 1:obj.labels
                    expValue = exp(obj.outputArray(hh,ii));
                    sum2 = sum2 + expValue;
                end
                for jj = 1:obj.labels
                    errorValue = YTrue(jj,ii).*(log(exp(obj.outputArray(jj,ii))./sum2));
                    sum1 = sum1 + errorValue;
                end
            end
            loss = -(sum1./obj.samples);

            %Inserimento del valore calcolato all'interno del vettore che
            %contiene tutti gli errori di perdita
            obj.lossVector = horzcat(obj.lossVector,loss);
        end

        function obj = accuracyFunction(obj,y,currentEpochs)
            %Funzione che calcola l'accuratezza della previsione della rete
            %neurale confrontando le uscite calcolate con quelle vere
            choiceY = [];
            for i = 1:obj.samples
                currentY = find(obj.outputArray(:,i) == max(obj.outputArray(:,i)),1,"first");
                choiceY(i) = obj.labelVector(currentY);
            end
            accuracy = (sum(y == transpose(choiceY)))./obj.samples;
            obj.accuracyVector = horzcat(obj.accuracyVector,accuracy);
            
            if (currentEpochs == obj.epochs)
                %Se ho studiato l'ultima epoca allora mi stampo il valore di
                %accuratezza finale, cioè quello a regime
                fprintf("Valore di accuratezza a regime:");
                disp(accuracy);
            end
        end

        function obj = forwardPropagation(obj,X)
            %Procedura di propagazione in avanti in cui dato un insieme di
            %campioni si calcola l'uscita associatovi
            
            %Propagazione in avanti del primo strato nascosto
            obj.a1 = obj.W1*X - obj.B1;
            obj.z1 = reluFunction(obj,obj.a1);
            %Propagazione in avanti del secondo strato nascosto
            obj.a2 = obj.W2*obj.z1 - obj.B2;
            obj.z2 = reluFunction(obj,obj.a2);
            %Propagazione in avanti delo strato di uscita
            obj.a3 = obj.W3*obj.z2 - obj.B3;

            if (obj.labels == 2)
                %Caso di classificazione binaria

                %Utilizzo la funzione di attivazione 'sigmoide'.
                %Infatti per problemi di classificazione binaria, la
                %funzione 'sigmoide' è preferibile alla funzione  'relu'
                %poichè permette di dare importanza anche ai valori
                %negativi ottenuti

                %A3 = dlarray(obj.a3,'CB');
                %Z3 = sigmoid(A3);
                %obj.z3 = extractdata(Z3);
                
                %Non utilizzo nessuna funzione di attivazione
                obj.z3 = obj.a3;
            else
                %Caso di classificazione multiclasse

                %Utilizzo la funzione di attivazione 'relu'.
                %In questo caso la funzione 'relu' è preferibile poichè
                %avendo più etichette di uscita si va a selezionare meglio
                %le etichette più probabili e scartare tutte le altre
                obj.z3 = reluFunction(obj,obj.a3);
            end

            %Applico la funzione 'softmax' all'uscita della rete neurale
            finalValue = softmax(obj.z3);

            %Inserimento dei nuovi valori calcolati all'interno del
            %vettore delle uscite predette
            obj.outputArray = horzcat(obj.outputArray,finalValue);
        end

        function obj = backPropagation(obj,X,y)
            %Procedura di back propagation in cui si calcola la derivata
            %della funzione di perdita rispetto alle matrici dei pesi così
            %da aggiornare queste ultime, lo stesso per i vettori di bias
            dEdb1 = zeros(obj.n(2),1);
            dEdb2 = zeros(obj.n(3),1);
            dEdb3 = zeros(obj.n(4),1);
            dEdw1 = zeros(obj.n(2),obj.features);
            dEdw2 = zeros(obj.n(end-1),obj.n(2));
            dEdw3 = zeros(obj.labels,obj.n(end-1));
            
            %Calcolo del vettore YTrue da utilizzare nella funzione
            %'derivativeComputation' per il calcolo di dEdy
            YTrue = zeros(obj.labels,size(y,1));
            for i = 1:size(y,1)
                index = find(obj.labelVector == y(i));
                YTrue(index,i) = 1;
            end

            for sample = 1:size(y,1)
                %Per lo strato di uscita
                for j = 1:obj.labels            %Per ogni neurone dello strato di uscita       
                    %Calcolo della derivata dEdy
                    dEdy(j) = derivativeComputation(obj,YTrue,j,sample);
                    %Calcolo della derivata dEda
                    dEda(j,obj.numHiddenLayer+1) = dEdy(j).*gradientFunction(obj,obj.a3(j,sample));

                    for i = 1:obj.n(end-1)      %Per ogni neurone dell'ultimo strato nascosto
                        %Calcolo della derivata dadw
                        dadw(j,i) = obj.z2(i,sample);
                        %Calcolo della derivata finale dEdw per lo strato
                        %di uscita
                        dEdw3(j,i) = dEdw3(j,i) + (dEda(j,obj.numHiddenLayer+1).*dadw(j,i));
                    end
                    %Calcolo della derivata finale per lo strato di uscita
                    %rispetto alle componenti di bias (i = 0)
                    dEdb3(j,1) = dEdb3(j,1) - dEda(j,obj.numHiddenLayer+1);
                end

                %Per tutti gli altri strati che non sono di uscita
                for j = 1:obj.n(end-1)          %Per ogni neurone dell'ultimo strato nascosto
                    sum = 0;
                    for h = 1:obj.n(end)
                        sum = sum + (dEda(h,obj.numHiddenLayer+1).*obj.B3(h));
                    end
                    for h = 1:obj.n(end)
                        sum = sum + (dEda(h,obj.numHiddenLayer+1).*obj.W3(h,j));
                    end
                    dEda(j,obj.numHiddenLayer) = gradientFunction(obj,obj.a2(j,sample)).*sum;

                    for i = 1:obj.n(end-2)
                        %Calcolo della derivata dadw
                        dadw(j,i) = obj.z1(i,sample);
                        %Calcolo della derivata finale dEdw per l'ultimo
                        %strato nascosto
                        dEdw2(j,i) = dEdw2(j,i) + (dEda(j,obj.numHiddenLayer).*dadw(j,i));
                    end
                    dEdb2(j,1) = dEdb2(j,1) - dEda(j,obj.numHiddenLayer);
                end

                for j = 1:obj.n(end-2)          %Per ogni neurone del primo strato nascosto
                    sum = 0;
                    for h = 1:obj.n(end-1)
                        sum = sum + dEda(h,obj.numHiddenLayer).*obj.B2(h);
                    end
                    for h = 1:obj.n(end-1)
                        sum = sum + (dEda(h,obj.numHiddenLayer).*obj.W2(h,j));
                    end
                    dEda(j,1) = gradientFunction(obj,obj.a1(j,sample)).*sum;

                    for i = 1:obj.features
                        %Calcolo della derivata dadw
                        dadw(j,i) = X(i,sample);
                        %Calcolo della derivata finale dEdw per il primo
                        %strato nascosto
                        dEdw1(j,i) = dEdw1(j,i) + (dEda(j,1).*dadw(j,i));
                    end
                    dEdb1(j,1) = dEdb1(j,1) - dEda(j,1);
                end
            end
            %Una volta analizzati tutti i campioni dell'iterazione si va ad
            %eseguire l'aggiornamento dei pesi e dei bias
            
            %Caso con utilizzo del constant step-size
            obj.W1 = obj.W1 - (obj.alpha.*(dEdw1./size(y,1)));
            obj.W2 = obj.W2 - (obj.alpha.*(dEdw2./size(y,1)));
            obj.W3 = obj.W3 - (obj.alpha.*(dEdw3./size(y,1)));
            obj.B1 = obj.B1 - (obj.alpha.*(dEdb1./size(y,1)));
            obj.B2 = obj.B2 - (obj.alpha.*(dEdb2./size(y,1)));
            obj.B3 = obj.B3 - (obj.alpha.*(dEdb3./size(y,1)));

            %Caso con utilizzo del diminishing step-size
        %     obj.W1 = obj.W1 - (obj.dss.*(dEdw1./size(y,1)));
        %     obj.W2 = obj.W2 - (obj.dss.*(dEdw2./size(y,1)));
        %     obj.W3 = obj.W3 - (obj.dss.*(dEdw3./size(y,1)));
        %     obj.B1 = obj.B1 - (obj.dss.*(dEdb1./size(y,1)));
        %     obj.B2 = obj.B2 - (obj.dss.*(dEdb2./size(y,1)));
        %     obj.B3 = obj.B3 - (obj.dss.*(dEdb3./size(y,1)));
        end

        function obj = weigthUpdate(obj,X,Y)
            %Funzione che una volta richiamata va a lanciare per un numero
            %finito di volte le funzioni "forward propagation" e "back
            %propagation", usate per l'aggiornamento dei pesi e quindi
            %l'addestramento della rete neurale
            for i = 1:obj.epochs
                fprintf('Epoca corrente :');
                disp(i);
                %Considero "m" campioni per volta
                for j = 1:obj.m:obj.samples
                    if ((j+obj.m-1) > obj.samples)
                        %Caso in cui l'ultimo set di campioni da analizzare
                        %ha dimensione più piccola del numero di campioni
                        %'m', quindi prendo tutti i rimanenti
                        XTraining = X(:,j:obj.samples);
                        YTraining = Y(j:obj.samples);
                    else
                        %Caso in cui si prende un set di campioni di
                        %dimensione 'm'
                        XTraining = X(:,j:j+obj.m-1);
                        YTraining = Y(j:j+obj.m-1);
                    end

                    %Richiamo delle funzioni "forwardPropagation" e
                    %"backPropagation" per l'aggiornamento dei pesi
                    obj = forwardPropagation(obj,XTraining);

                    %Nel caso si usasse il diminishing step-size allora
                    %aggiorno tale parametro ad ogni fine epoca come segue

                    % obj.dss = 1./(1 + i);
                    obj = backPropagation(obj,XTraining,YTraining);
                end
                %Finisce un'epoca e vado ad effettuare il calcolo
                %dell'accuratezza e del valore di perdita e successivamente
                %resetto la variabile obj.outputArray
                obj = accuracyFunction(obj,Y,i);
                obj = errorFunction(obj,Y);

                if (i ~= obj.epochs)
                    %Reset del vettore delle uscite predette solo se non si
                    %è arrivati all'ultima epoca, contrariamente non viene
                    %resettata per analizzare tale vettore nel workspace
                    obj.outputArray = [];
                end
            end
        end

        function obj = accuracyTest(obj,X,Y)
            %Funzione che viene richiamata dopo la procedura di
            %aggiornamento dei pesi per adnare a calcolare l'accuratezza
            %sui campioni del Test Set usando le  matrici W ed i vettori B
            %appena calcolati
            obj.outputArray = [];               %Inizializzo la matrice vuota per i nuovi risultati
            obj.samples = size(X,2);            %Inizializzo il numero di campioni del Test Set
            obj = forwardPropagation(obj,X);

            %Nella successiva funzione inserisco come terzo parametro di
            %ingresso il numero di epoche (anche se inutile ai fini del
            %calcolo) per stampare il valore di accuratezza a schermo
            obj = accuracyFunction(obj,Y,obj.epochs);       
        end
    end
end