###############################################################################
#                                                                             #
#   STRASSEN_TESTING.PY                                                       #
#                                                                             #
#   Script per il benchmark delle tre varianti dell'algoritmo di Strassen     #
#   implementate in strassen_algorithms.py, confrontate con il prodotto       #
#   riga per colonna di NumPy (numpy.matmul).                                 #
#                                                                             #
#   Cosa misura                                                               #
#   -----------                                                               #
#   Ogni funzione di questo file genera un campione di coppie di matrici con   #
#   certe caratteristiche, cronometra i quattro algoritmi su ciascuna coppia   #
#   e scrive i tempi in un file CSV dentro la cartella dataset/. I sei        #
#   esperimenti previsti sono:                                                #
#                                                                             #
#     - different_q()         effetto del crossover point Q                   #
#     - sparse_dense_matrix() matrici sparse contro matrici dense             #
#     - lunghe_vs_larghe()    matrici con righe e colonne sbilanciate         #
#     - increasing_dimension() andamento al crescere delle dimensioni         #
#     - floating_time()       numeri reali contro numeri interi               #
#     - floating_error()      campione per l'analisi dell'errore numerico     #
#                                                                             #
#   Formato dei file prodotti                                                 #
#   -------------------------                                                 #
#   Tutti i CSV hanno la stessa intestazione:                                 #
#                                                                             #
#     ID,tipo,Q,a,b,c,riga_per_colonna,dynamic_time,static_time,peeling_time  #
#                                                                             #
#   dove ID identifica la coppia di matrici, "tipo" il gruppo sperimentale,   #
#   Q il crossover point, a/b/c le tre dimensioni (A e' [a x b], B e'         #
#   [b x c]) e le ultime quattro colonne i tempi in secondi dei quattro       #
#   algoritmi. Questo tracciato e' pensato per essere letto direttamente da   #
#   grafici.Rmd, che produce le figure del report.                            #
#                                                                             #
#   Nota sui tempi                                                            #
#   --------------                                                            #
#   Si usa time.process_time(), che misura il tempo di CPU del processo e non  #
#   il tempo "da orologio": e' quindi insensibile ad altri processi in        #
#   esecuzione sulla macchina. Il valore viene poi portato a un minimo di     #
#   1e-6 secondi, perche' su Windows la risoluzione del contatore e' grossa   #
#   (circa 1/64 di secondo) e le misure piu' rapide risulterebbero pari a     #
#   zero, valore scomodo da trattare nelle analisi successive.                #
#                                                                             #
#   Riproducibilita'                                                          #
#   ----------------                                                          #
#   Ogni esperimento fissa random.seed(1234) prima di generare le dimensioni, #
#   cosi' ripetendo l'esecuzione si ottengono le stesse forme di matrici.     #
#                                                                             #
#   Uso                                                                       #
#   ---                                                                       #
#   Si lancia lo script con la cartella del progetto come directory corrente, #
#   dopo aver scelto nel blocco finale (__main__) quale esperimento eseguire. #
#   La cartella dataset/ deve esistere.                                       #
#                                                                             #
#   Autore: Luca Varotto                                                      #
#   Esame di Sistemi di Elaborazione 2 - Universita' degli Studi di Padova    #
#                                                                             #
###############################################################################

import strassen_algorithms as s_a   #le tre varianti di Strassen da testare
import numpy as nmp  #per l'implementazione delle matrici
import random        #per generare i numeri casuali
import time          #per tracciare i tempi

#############################################################################
################## FUNZIONI PER IL TESTING DEGLI ALGORITMI ##################
#############################################################################

# Genera la coppia di matrici su cui eseguire il prodotto.
## Argomenti:
## @a, il numero di righe della prima matrice
## @b, le colonne della prima matrice e le righe della seconda
## @c, il numero di colonne della seconda matrice
### Return:
### @A, @B, le due matrici di interi casuali
#
# Le celle contengono interi casuali tra -100 e 99; il tipo 'l' (long)
# evita che i prodotti, che possono crescere molto, vadano in overflow.
# Le dimensioni interne coincidono per costruzione, quindi il prodotto
# A*B e' sempre definito e vale [a x c].
def generate_2_random_matrix(a,b,c):
    A = nmp.random.randint(low=-100, high=100, size=(a,b), dtype='l')
    B = nmp.random.randint(low=-100, high=100, size=(b,c), dtype='l')
    return A, B

# Primo esperimento: come cambiano le prestazioni al variare del
# crossover point Q.
## Argomenti:
## @dim_min, la dimensione minima delle matrici generate
## @dim_max, la dimensione massima delle matrici generate
## @n_test, il numero di coppie di matrici da generare
## @file_name, il percorso del file CSV in cui scrivere i tempi
## @qmin, @qmax, gli esponenti minimo e massimo per Q (Q vale 2**q)
#
# E' un disegno a misure ripetute: la stessa coppia di matrici viene
# riutilizzata per tutti i valori di Q, cosi' il confronto non e' disturbato
# dalla variabilita' delle dimensioni. Il prodotto riga per colonna si
# cronometra una volta sola, non dipendendo da Q, e il suo tempo viene
# ripetuto su ogni riga del file.
def different_q(dim_min: int, dim_max: int, n_test:int, file_name:str, qmin, qmax):

    random.seed(1234)   #seed fissato: dimensioni riproducibili

    with open(file_name, "w") as f:

        #intestazione del CSV, comune a tutti gli esperimenti
        f.write(f"ID,tipo,Q,a,b,c,riga_per_colonna,dynamic_time,static_time,peeling_time\n")

        for i in range(0,n_test):

            #le tre dimensioni sono estratte in modo indipendente, quindi
            #le matrici sono genericamente rettangolari
            a = random.randint(dim_min,dim_max)
            b = random.randint(dim_min,dim_max)
            c = random.randint(dim_min,dim_max)

            A, B = generate_2_random_matrix(a,b,c)

            print("\n-----------Matrice numero", i+1, end="-----------\n")

            #riferimento: il prodotto riga per colonna di NumPy
            print("Prodotto riga per colonna...")
            start = time.process_time()
            nmp.matmul(A,B)
            end = time.process_time()
            rbc_time = max(end - start, 1e-6)   #minimo 1e-6 per evitare gli zeri

            #sulla stessa coppia si prova ogni Q della griglia
            for q in range(qmin,qmax):
                Q = 2**q   #i valori di Q sono potenze di 2

                print(f"\n---Q={Q}---")
                #prima parte della riga: identificativi e tempo di riferimento
                f.write(f"{i},rettangolare,{Q},{a},{b},{c},{rbc_time},")
                #la seconda parte, con i tre tempi di Strassen, la completa
                #Strassens_speed_test()
                Strassens_speed_test(A=A,B=B,f=f, Q=(Q))

        f.close()


# Genera una matrice sparsa, ossia con molte celle nulle.
## Argomenti:
## @rows, il numero di righe
## @cols, il numero di colonne
### Return:
### @ris, la matrice sparsa
#
# Esattamente metà delle celle viene riempita con valori casuali, l'altra
# metà resta a zero. Le posizioni dei valori non nulli sono scelte senza
# ripetizioni (replace=False) tra tutte quelle disponibili, e vengono
# riempite tramite l'attributo .flat, che vede la matrice come un unico
# vettore. A differenza di generate_2_random_matrix() il risultato e' di
# tipo float, perche' nasce da nmp.zeros().
def generate_sparse_matrix(rows, cols):

    total_elements = rows * cols                        #celle totali
    num_nonzero_elements = int(total_elements * 0.5)    #meta' non nulle

    random_values = nmp.random.randint(low=-100, high=100, size=num_nonzero_elements, dtype='l')

    ris = nmp.zeros((rows, cols))   #si parte da una matrice di soli zeri
    #scelta delle posizioni da riempire, senza ripetizioni
    indices = nmp.random.choice(range(total_elements), size=num_nonzero_elements, replace=False)
    ris.flat[indices] = random_values   #inserimento dei valori casuali

    return ris


# Secondo esperimento: matrici dense contro matrici sparse.
## Argomenti:
## @dim_min, @dim_max, l'intervallo delle dimensioni
## @Q, il crossover point, fissato per tutto l'esperimento
## @n_test, il numero di coppie per ciascuno dei due gruppi
#
# Vengono generati due campioni della stessa numerosita': nel primo le
# matrici sono piene di interi casuali ("densa"), nel secondo metà delle
# celle e' nulla ("sparsa"). La colonna "tipo" del CSV distingue i due
# gruppi, e gli ID del secondo gruppo proseguono la numerazione del primo
# (n_test+i) perche' si tratta di coppie di matrici diverse.
def sparse_dense_matrix(dim_min: int, dim_max: int, Q:int, n_test:int):

    random.seed(1234)

    with open("dataset/sparse_vs_dense.txt", "w") as f:

        f.write(f"ID,tipo,Q,a,b,c,riga_per_colonna,dynamic_time,static_time,peeling_time\n")

        #---------------------- primo gruppo: matrici dense ----------------------
        print("Matrici dense")
        for i in range(0,n_test):

            a = random.randint(dim_min, dim_max)
            b = random.randint(dim_min, dim_max)
            c = random.randint(dim_min, dim_max)

            A, B = generate_2_random_matrix(a,b,c)   #matrici piene

            print("\n-----------Matrice numero", i+1, end="-----------\n")

            print("Prodotto riga per colonna...")
            start = time.process_time()
            nmp.matmul(A,B)
            end = time.process_time()
            rbc_time = max(end - start, 1e-6)

            f.write(f"{i},densa,{Q},{a},{b},{c},{rbc_time},")

            Strassens_speed_test(A=A,B=B,f=f, Q=(Q))

        #--------------------- secondo gruppo: matrici sparse --------------------
        print("Matrici sparse")
        for i in range(0,n_test):

            a = random.randint(dim_min, dim_max)
            b = random.randint(dim_min, dim_max)
            c = random.randint(dim_min, dim_max)

            #qui le due matrici vengono costruite separatamente, curando che
            #le dimensioni interne restino compatibili: A e' [a x b], B e' [b x c]
            A = generate_sparse_matrix(a,b)
            B = generate_sparse_matrix(b,c)

            print("\n-----------Matrice numero", i+1, end="-----------\n")

            print("Prodotto riga per colonna...")
            start = time.process_time()
            nmp.matmul(A,B)
            end = time.process_time()
            rbc_time = max(end - start, 1e-6)

            #gli ID ripartono da n_test per non sovrapporsi a quelli delle dense
            f.write(f"{n_test+i},sparsa,{Q},{a},{b},{c},{rbc_time},")

            Strassens_speed_test(A=A,B=B,f=f, Q=(Q))

        f.close()


# Terzo esperimento: matrici "lunghe" e "larghe", ossia molto sbilanciate.
## Argomenti:
## @dim_min, @dim_max, l'intervallo delle dimensioni "grandi"
## @Q, il crossover point, fissato per tutto l'esperimento
## @n_test, il numero di coppie per ciascuno dei tre gruppi
#
# Una matrice si dice "lunga" quando ha molte piu' righe che colonne e
# "larga" nel caso opposto. Lo sbilanciamento si ottiene dividendo per 10
# l'intervallo di una delle tre dimensioni, una per gruppo:
#
#   - gruppo "a_piccolo": a e' un decimo, quindi A e' larga;
#   - gruppo "b_piccolo": b e' un decimo, quindi A e' lunga e B e' larga;
#   - gruppo "c_piccolo": c e' un decimo, quindi B e' lunga.
#
# Il confronto e' interessante perche' le varianti usano criteri diversi
# per il crossover point: la dimensione minima per lo static padding, la
# media geometrica per le altre due. Con dimensioni molto sbilanciate i due
# criteri portano a profondita' di ricorsione diverse.
def lunghe_vs_larghe(dim_min: int, dim_max: int, Q:int, n_test:int):

    random.seed(1234)

    with open("dataset/lunghe_vs_larghe.txt", "w") as f:

        f.write(f"ID,tipo,Q,a,b,c,riga_per_colonna,dynamic_time,static_time,peeling_time\n")

        #-------------------- primo gruppo: A larga (a piccolo) ------------------
        print("########## a piccolo ##########")
        for i in range(0,n_test):

            a = random.randint(dim_min//10, dim_max//10)   #un decimo delle altre
            b = random.randint(dim_min, dim_max)
            c = random.randint(dim_min, dim_max)

            A, B = generate_2_random_matrix(a,b,c)

            print("\n-----------Matrice numero", i+1, end="-----------\n")

            print("Prodotto riga per colonna...")
            start = time.process_time()
            nmp.matmul(A,B)
            end = time.process_time()
            rbc_time = max(end - start, 1e-6)

            f.write(f"{i},a_piccolo,{Q},{a},{b},{c},{rbc_time},")

            Strassens_speed_test(A=A,B=B,f=f, Q=(Q))


        #------------ secondo gruppo: A lunga e B larga (b piccolo) --------------
        print("\n########## b piccolo ##########")
        for i in range(0,n_test):

            a = random.randint(dim_min, dim_max)
            b = random.randint(dim_min//10, dim_max//10)   #la dimensione interna
            c = random.randint(dim_min, dim_max)

            A, B = generate_2_random_matrix(a,b,c)

            print("\n-----------Matrice numero", i+1, end="-----------\n")

            print("Prodotto riga per colonna...")
            start = time.process_time()
            nmp.matmul(A,B)
            end = time.process_time()
            rbc_time = max(end - start, 1e-6)

            f.write(f"{i},b_piccolo,{Q},{a},{b},{c},{rbc_time},")

            Strassens_speed_test(A=A,B=B,f=f, Q=(Q))



        #------------------- terzo gruppo: B lunga (c piccolo) -------------------
        print("\n########## c piccolo ##########")
        for i in range(0,n_test):

            a = random.randint(dim_min, dim_max)
            b = random.randint(dim_min, dim_max)
            c = random.randint(dim_min//10, dim_max//10)   #poche colonne in B

            A, B = generate_2_random_matrix(a,b,c)

            print("\n-----------Matrice numero", i+1, end="-----------\n")

            print("Prodotto riga per colonna...")
            start = time.process_time()
            nmp.matmul(A,B)
            end = time.process_time()
            rbc_time = max(end - start, 1e-6)

            f.write(f"{i},c_piccolo,{Q},{a},{b},{c},{rbc_time},")

            Strassens_speed_test(A=A,B=B,f=f, Q=(Q))

        f.close()



# Quarto esperimento: andamento dei tempi al crescere delle dimensioni.
## Argomenti:
## @dim_min0, @dim_max0, l'intervallo iniziale delle dimensioni
## @Q, il crossover point, fissato per tutto l'esperimento
## @n_test, il numero di coppie da generare
#
# A differenza degli altri esperimenti l'intervallo non e' fisso: ad ogni
# iterazione i due estremi vengono spostati di 10, cosi' le coppie generate
# sono via via piu' grandi e il campione copre una fascia ampia di
# dimensioni. In grafici.Rmd ogni coppia viene poi rappresentata in base
# alla media geometrica di a, b e c, che riassume la "taglia" del prodotto.
def increasing_dimension(dim_min0: int, dim_max0: int, Q:int, n_test:int):

    random.seed(1234)

    with open("dataset/dimensioni.txt", "w") as f:

        f.write(f"ID,tipo,Q,a,b,c,riga_per_colonna,dynamic_time,static_time,peeling_time\n")

        for i in range(0,n_test):

            #l'intervallo si sposta verso l'alto ad ogni iterazione
            dim_min0 += 10
            dim_max0 += 10

            a = random.randint(dim_min0, dim_max0)
            b = random.randint(dim_min0, dim_max0)
            c = random.randint(dim_min0, dim_max0)

            A, B = generate_2_random_matrix(a,b,c)

            print("\n-----------Matrice numero", i+1, end="-----------\n")

            print("Prodotto riga per colonna...")
            start = time.process_time()
            nmp.matmul(A,B)
            end = time.process_time()
            rbc_time = max(end - start, 1e-6)

            f.write(f"{i},dimensioni_crescenti,{Q},{a},{b},{c},{rbc_time},")

            Strassens_speed_test(A=A,B=B,f=f, Q=(Q))


# Quinto esperimento: matrici di numeri reali invece che di interi.
## Argomenti:
## @dim_min0, @dim_max0, l'intervallo delle dimensioni
## @Q, il crossover point, fissato per tutto l'esperimento
## @n_test, il numero di coppie da generare
#
# Le celle sono riempite con nmp.random.uniform(), quindi con numeri in
# virgola mobile anziche' interi. Il confronto si fa con il gruppo delle
# matrici dense di sparse_dense_matrix(), generato con le stesse dimensioni
# e lo stesso seed: le due colonne "tipo" ("floating" e "densa") vengono
# unite in grafici.Rmd.
def floating_time(dim_min0: int, dim_max0: int, Q:int, n_test:int):

    random.seed(1234)

    with open("dataset/on_floating.txt", "w") as f:

        f.write(f"ID,tipo,Q,a,b,c,riga_per_colonna,dynamic_time,static_time,peeling_time\n")
        for i in range(0,n_test):

            a = random.randint(dim_min0, dim_max0)
            b = random.randint(dim_min0, dim_max0)
            c = random.randint(dim_min0, dim_max0)

            #numeri reali uniformi tra -100 e 100
            A = nmp.random.uniform(low=-100, high=100, size=(a,b))
            B = nmp.random.uniform(low=-100, high=100, size=(b,c))

            print("\n-----------Matrice numero", i+1, end="-----------\n")

            print("Prodotto riga per colonna...")
            start = time.process_time()
            nmp.matmul(A,B)
            end = time.process_time()
            rbc_time = max(end - start, 1e-6)

            f.write(f"{i},floating,{Q},{a},{b},{c},{rbc_time},")

            Strassens_speed_test(A=A,B=B,f=f, Q=(Q))


# Sesto esperimento: campione dedicato all'instabilita' numerica.
## Argomenti:
## @dim_min0, @dim_max0, l'intervallo delle dimensioni
## @Q, il crossover point, fissato per tutto l'esperimento
## @n_test, il numero di coppie da generare
## @floating_accuracy, il numero di decimali a cui arrotondare i confronti
#
# Serve il campione in virgola mobile su cui studiare quanto i risultati di
# Strassen si discostano da quelli del prodotto riga per colonna: le somme
# e sottrazioni in piu' richieste dall'algoritmo accumulano errori di
# arrotondamento che il prodotto classico non ha. Il confronto vero e
# proprio tra le matrici risultato si ottiene con Strassens_results().
def floating_error(dim_min0: int, dim_max0: int, Q:int, n_test:int, floating_accuracy):

    random.seed(1234)

    with open("dataset/error_floating.txt", "w") as f:

        f.write(f"ID,tipo,Q,a,b,c,riga_per_colonna,dynamic_time,static_time,peeling_time\n")

        for i in range(0,n_test):

            a = random.randint(dim_min0, dim_max0)
            b = random.randint(dim_min0, dim_max0)
            c = random.randint(dim_min0, dim_max0)

            A = nmp.random.uniform(low=-100, high=100, size=(a,b))
            B = nmp.random.uniform(low=-100, high=100, size=(b,c))

            print("\n-----------Matrice numero", i+1, end="-----------\n")
            print("Prodotto riga per colonna...")
            start = time.process_time()
            nmp.matmul(A,B)
            end = time.process_time()
            rbc_time = max(end - start, 1e-6)

            f.write(f"{i},floating,{Q},{a},{b},{c},{rbc_time},")

            Strassens_speed_test(A=A,B=B,f=f, Q=(Q))



# Cronometra le tre varianti di Strassen su una coppia di matrici e scrive
# i tempi nel file, completando la riga del CSV.
## Argomenti:
## @A, @B, la coppia di matrici su cui eseguire il prodotto
## @f, il file CSV aperto in scrittura
## @Q, il crossover point da usare
#
# E' la funzione usata da tutti gli esperimenti: ognuno scrive la prima
# parte della riga (identificativi, dimensioni e tempo del riga per
# colonna) e delega a questa la parte finale, terminata dal ritorno a capo.
# L'ordine delle tre colonne e' sempre dynamic, static, peeling, coerente
# con l'intestazione. I risultati dei prodotti non vengono conservati:
# qui interessa solo il tempo impiegato.
def Strassens_speed_test(A,B,f,Q):

    #------------------------------ dynamic padding --------------------------
    print("Strassen padding dinamico...")

    start = time.process_time()
    s_a.Strassen_dynamic_padding(A,B,Q)
    end = time.process_time()
    dynamic_time = max(end - start, 1e-6)

    #------------------------------- static padding --------------------------
    print("Strassen padding statico...")

    start = time.process_time()
    s_a.Strassen_static_padding(A,B,Q)
    end = time.process_time()
    static_time = max(end - start, 1e-6)

    #------------------------------ dynamic peeling --------------------------
    print("Strassen peeling dinamico...")

    start = time.process_time()
    s_a.Strassen_dynamic_peeling(A,B,Q)
    end = time.process_time()
    peeling_time = max(end - start, 1e-6)

    #ultime tre colonne della riga, con il ritorno a capo finale
    f.write(f"{dynamic_time},{static_time},{peeling_time}\n")


# Calcola il prodotto A*B con tutti e quattro gli algoritmi e restituisce
# le quattro matrici risultato.
## Argomenti:
## @A, @B, la coppia di matrici su cui eseguire il prodotto
## @Q, il crossover point da usare
### Return:
### @C_RBC, il risultato del prodotto riga per colonna
### @C_DPA, il risultato del dynamic padding
### @C_SPA, il risultato dello static padding
### @C_DPE, il risultato del dynamic peeling
#
# Non misura tempi: serve a confrontare i risultati tra loro, prendendo
# quello del prodotto riga per colonna come riferimento. Con matrici di
# interi le quattro matrici coincidono esattamente, mentre in virgola
# mobile emergono le differenze dovute agli arrotondamenti.
def Strassens_results(A,B,Q):

    C_RBC = nmp.matmul(A,B)                         #riferimento
    C_DPA = s_a.Strassen_dynamic_padding(A,B,Q)     #dynamic padding
    C_SPA = s_a.Strassen_static_padding(A,B,Q)      #static padding
    C_DPE = s_a.Strassen_dynamic_peeling(A,B,Q)     #dynamic peeling

    return C_RBC, C_DPA, C_SPA, C_DPE


#############################################################################


# Blocco eseguito solo lanciando questo file direttamente (e non quando
# viene importato da un altro modulo): si attiva un esperimento alla volta,
# lasciando commentati gli altri, perche' ognuno sovrascrive il proprio
# file in dataset/ e richiede diversi minuti di calcolo.
#
# Gli argomenti ricorrenti sono, nell'ordine: l'intervallo delle dimensioni,
# il crossover point Q e la numerosita' del campione.

if __name__ == "__main__":

    #confronto tra i valori di Q: da 2**4=16 a 2**9=512, su 30 coppie
    different_q(600, 1000, n_test=30, file_name="dataset/q_test.txt", qmin=4, qmax=10)
    #matrici dense contro matrici sparse, 40 coppie per gruppo
    #sparse_dense_matrix(600, 1000, 128, 40)
    #matrici sbilanciate, 60 coppie per ciascuno dei tre gruppi
    #lunghe_vs_larghe(800, 1200, 64, 60)
    #dimensioni crescenti, 75 coppie a partire dall'intervallo 200-400
    #increasing_dimension(200,400,128,75)
    #matrici di numeri reali, 40 coppie
    #floating_time(600, 1000, 128, 40)
    #campione per l'errore numerico, confronti arrotondati a 8 decimali
    #floating_error(600, 1000, 128, 40, 8)