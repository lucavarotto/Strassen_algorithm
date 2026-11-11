###############################################################################
#                                                                             #
#   STRASSEN_ALGORITHMS.PY                                                    #
#                                                                             #
#   Implementazione in Python dell'algoritmo di Strassen per il prodotto      #
#   matriciale e di tre sue estensioni per matrici di dimensioni qualsiasi.   #
#                                                                             #
#   Contesto                                                                  #
#   --------                                                                  #
#   Nel 1969 V. Strassen mostro' che il prodotto tra due matrici [2 x 2] si   #
#   puo' calcolare con 7 moltiplicazioni invece delle 8 richieste dal         #
#   classico prodotto riga per colonna. Applicando ricorsivamente questa      #
#   idea alle sottomatrici (i "blocchi") si passa da una complessita'         #
#   O(n^3) a O(n^log2(7)) = O(n^2.807).                                       #
#                                                                             #
#   Idea di fondo                                                             #
#   ------------                                                              #
#   Date A e B partizionate in quattro blocchi ciascuna                       #
#                                                                             #
#       A = | A11  A12 |        B = | B11  B12 |                              #
#           | A21  A22 |            | B21  B22 |                             #
#                                                                             #
#   si calcolano i sette prodotti                                            #
#                                                                             #
#       M1 = (A11 + A22)(B11 + B22)                                          #
#       M2 = (A21 + A22)(B11)                                                #
#       M3 = (A11)(B12 - B22)                                                #
#       M4 = (A22)(B21 - B11)                                                #
#       M5 = (A11 + A12)(B22)                                                #
#       M6 = (A21 - A11)(B11 + B12)                                          #
#       M7 = (A12 - A22)(B21 + B22)                                          #
#                                                                             #
#   e si ricompone il risultato come                                          #
#                                                                             #
#       C11 = M1 + M4 - M5 + M7       C12 = M3 + M5                          #
#       C21 = M2 + M4                 C22 = M1 - M2 + M3 + M6                #
#                                                                             #
#   Il problema delle dimensioni dispari                                      #
#   -----------------------------------                                       #
#   Il partizionamento in quattro blocchi di uguali dimensioni richiede che   #
#   righe e colonne siano in numero pari: con una dimensione dispari la       #
#   divisione a meta' non e' definita. Questo file implementa le tre          #
#   strategie proposte in letteratura per aggirare il problema:               #
#                                                                             #
#     - DYNAMIC PADDING: ad ogni chiamata ricorsiva, se una dimensione e'     #
#       dispari, si aggiunge una riga/colonna di zeri;                        #
#     - STATIC PADDING: le righe/colonne di zeri necessarie vengono           #
#       aggiunte una sola volta, alla prima chiamata, in quantita'            #
#       sufficiente a garantire dimensioni pari in tutta la ricorsione;       #
#     - DYNAMIC PEELING: ad ogni chiamata, se una dimensione e' dispari, si   #
#       "sbuccia" (peel) l'ultima riga/colonna, che viene poi trattata a      #
#       parte con la teoria delle matrici a blocchi.                          #
#                                                                             #
#   Il "crossover point" Q                                                    #
#   ----------------------                                                    #
#   La ricorsione di Strassen conviene solo finche' le matrici sono grandi:   #
#   sotto una certa soglia il costo delle somme, delle copie e delle          #
#   chiamate di funzione supera il risparmio di una moltiplicazione.          #
#   Il parametro Q indica quella soglia: raggiunta, si delega il prodotto a   #
#   numpy.matmul(). Le diverse funzioni usano criteri diversi per decidere    #
#   quando Q e' stato raggiunto (dimensione minima o media geometrica delle   #
#   dimensioni), ed e' uno dei fattori che ne spiega le prestazioni           #
#   differenti.                                                               #
#                                                                             #
#   Convenzioni usate nei commenti                                           #
#   ------------------------------                                            #
#   Nel seguito A ha dimensioni [RA x CA], B ha dimensioni [RB x CB] con      #
#   RB = CA, e il risultato C ha dimensioni [RA x CB].                        #
#                                                                             #
#   Le funzioni di test e i benchmark si trovano in strassen_testing.py.      #
#                                                                             #
#   Autore: Luca Varotto                                                      #
#   Esame di Sistemi di Elaborazione 2 - Universita' degli Studi di Padova    #
#                                                                             #
###############################################################################

import numpy as nmp  #per l'implementazione delle matrici
import math          #per i logaritmi e arrotondare i numeri

#############################################################################
###################### ALGORITMO DI STRASSEN ORIGINALE ######################
#############################################################################

# In questa prima sezione si trova la versione "originale" dell'algoritmo,
# quella piu' vicina al paper del 1969, insieme alle funzioni ausiliarie
# per partizionare e ricomporre le matrici a blocchi. Queste ultime vengono
# riutilizzate da tutte le varianti definite piu' avanti.

# Funzione per dividere una matrice in 4 blocchi di uguali dimensioni
## Argomenti:
## @M, la matrice
### Return:
### @M11, @M12, @M21, @M22, ossia i 4 blocchi
#
# Il taglio avviene a meta' delle righe e a meta' delle colonne. La
# divisione intera (//) individua l'indice centrale; lo slicing di NumPy
# non copia i dati ma restituisce delle "viste" sulla matrice originale,
# quindi questa operazione e' molto economica in termini di memoria.
def split_matrix(M):
    mid_r = M.shape[0] // 2   #indice di meta' delle righe
    mid_c = M.shape[1] // 2   #indice di meta' delle colonne
    #ordine dei blocchi restituiti: alto-sinistra, alto-destra,
    #basso-sinistra, basso-destra
    return M[:mid_r, :mid_c], M[:mid_r, mid_c:], M[mid_r:, :mid_c], M[mid_r:, mid_c:]

# Funzione per unire una matrice divisa in 4 blocchi
## Argomenti:
## @M11, @M12, @M21, @M22, ossia i 4 blocchi
### Return:
### @M, la matrice
#
# E' l'operazione inversa di split_matrix(): prima si affiancano i blocchi
# orizzontalmente (hstack) due a due, ottenendo la riga superiore e quella
# inferiore, poi si impilano verticalmente (vstack) i due risultati.
def merge_matrix(M11, M12, M21, M22):
    return nmp.vstack((nmp.hstack((M11, M12)), nmp.hstack((M21, M22))))

# Funzione che riceve i 7 prodotti indicati da strassen e li somma per
# trovare i blocchi della matrice C
## Argomenti:
## @M1, @M2, @M3, @M4, @M5, @M6, @M7, le matrici ottenute con i prodotti
### Return:
### @C11, @C12, @C21, @C22, ossia i 4 blocchi della matrice dei risultati
#
# Sono le quattro combinazioni lineari riportate nell'intestazione del file:
# e' qui che si "paga" il risparmio di una moltiplicazione con qualche
# somma in piu', operazione pero' di costo quadratico e non cubico.
def strassen_sums(M1, M2, M3, M4, M5, M6, M7):
    return (M1 + M4 - M5 + M7), (M3 + M5), (M2 + M4), (M1 - M2 + M3 + M6)

# Funzione ricorsiva che esegue la versione
# di base dell'algoritmo di Strassen.
# Accetta quindi 2 matrici di pari dimensioni
# con numero di righe e colonne potenze di 2
## Argomenti:
## @A: la prima matrice della moltiplicazione
## @B: la seconda matrice della moltiplicazione
## @Q: livello a cui fermare la ricorsione
### Return:
### @C: la matrice risultante
#
# E' la versione di riferimento, quella che segue il paper originale: non
# prevede alcuna strategia per le dimensioni dispari, quindi e' applicabile
# solo a matrici quadrate le cui dimensioni restano pari ad ogni divisione
# (tipicamente potenze di 2). Le tre varianti delle sezioni successive
# nascono proprio per superare questo limite.
def Strassen_squareM(A,B,Q=2):

    #se si supera il crossover point si passa al prodotto riga per colonna
    #(caso base della ricorsione: sotto la soglia Q conviene numpy.matmul)
    if A.size[0] <= Q:
        return nmp.matmul(A,B)

    #se una dimensione è dispari non funziona
    #(il partizionamento in blocchi di uguali dimensioni non sarebbe definito)
    if A.size[0]%2==1:
        raise ValueError("Funziona solo con matrici di dimensioni pari")

    #partizionamento in quattro blocchi di uguali dimensioni
    A11, A12, A21, A22 = split_matrix(B)
    B11, B12, B21, B22 = split_matrix(B)

    #i sette prodotti di Strassen: ciascuno e' una nuova chiamata ricorsiva
    #su matrici di lato dimezzato
    M1 = Strassen_squareM(A11+A22, B11+B22)
    M2 = Strassen_squareM(A21+A22, B11)
    M3 = Strassen_squareM(A11, B12-B22)
    M4 = Strassen_squareM(A22, B21-B11)
    M5 = Strassen_squareM(A11+A12, B22)
    M6 = Strassen_squareM(A21-A11, B11+B12)
    M7 = Strassen_squareM(A12-A22, B21+B22)

    #dai sette prodotti si ricavano i quattro blocchi del risultato
    C11, C12, C21, C22 = strassen_sums(M1, M2, M3, M4, M5, M6, M7)

    #ricomposizione dei blocchi nella matrice finale
    C = merge_matrix(C11, C12, C21, C22)

    return C

#############################################################################
############## ALGORITMO DI STRASSEN PER MATRICI RETTANGOLARI ###############
#############################################################################

# Variante che accetta matrici rettangolari, non necessariamente quadrate.
# Resta pero' il vincolo sulle dimensioni pari: per questo non viene usata
# da sola, ma come "motore" dell'approccio static padding, che si occupa di
# garantire a priori il rispetto di quel vincolo.

# Funzione ricorsiva che esegue la versione
# di base dell'algoritmo di Strassen su matrici rettangolari.
# Accetta quindi 2 matrici di dimensioni pari e che rimangono
# pari ad ogni divisione fino al raggiungimento del crossover point
## Argomenti:
## @A: la prima matrice della moltiplicazione
## @B: la seconda matrice della moltiplicazione
## @Q: livello a cui fermare la ricorsione
## Return:
## @C: la matrice risultante
def Strassen_rectM(A,B,Q):

    a, b = A.shape #dimensioni delle matrici quadrate
    c = B.shape[1] #numero di colonne di B: il risultato sara' [a x c]

    # se viene raggiunto il cross-over point si passa al riga per colonna
    # qui il criterio e' la dimensione piu' piccola tra le tre: basta che
    # una sola scenda sotto Q perche' la ricorsione si fermi
    if a <= Q or b <= Q or c <= Q:
        return nmp.matmul(A,B)

    #se una dimensione è dispari non funziona
    #spetta al chiamante (static padding) garantire dimensioni sempre pari
    if a%2==1 or b%2==1 or c%2==1:
        raise ValueError("Funziona solo con matrici di dimensioni pari")

    # partizione di A e B in 4 blocchi di uguali dimensioni
    A11, A12, A21, A22 = split_matrix(A)
    B11, B12, B21, B22 = split_matrix(B)

    #calcolo dei prodotti indicati da strassen
    #ad ogni livello il parametro Q viene propagato, cosi' tutti i rami
    #della ricorsione si fermano alla stessa soglia
    M1 = Strassen_rectM(A11+A22, B11+B22,Q)
    M2 = Strassen_rectM(A21+A22, B11,Q)
    M3 = Strassen_rectM(A11, B12-B22,Q)
    M4 = Strassen_rectM(A22, B21-B11,Q)
    M5 = Strassen_rectM(A11+A12, B22,Q)
    M6 = Strassen_rectM(A21-A11, B11+B12,Q)
    M7 = Strassen_rectM(A12-A22, B21+B22,Q)

    #calcolo delle somme indicate da strassen
    C11, C12, C21, C22 = strassen_sums(M1, M2, M3, M4, M5, M6, M7)

    #unione dei risultati
    C = merge_matrix(C11, C12, C21, C22)

    return C

#############################################################################
######################## ALGORITMO "DYNAMIC PADDING" ########################
#############################################################################

# Primo dei tre approcci per le dimensioni dispari, proposto da Strassen
# stesso. La correzione e' fatta "al volo": ad ogni chiamata ricorsiva si
# guardano le dimensioni correnti e, se dispari, si aggiunge una sola
# riga/colonna di zeri, il minimo indispensabile per poter partizionare.
#
# Perche' gli zeri non falsano il risultato:
#   - una riga di zeri aggiunta in fondo ad A produce una riga di zeri in
#     fondo a C, che va quindi rimossa prima di restituire il risultato;
#   - una colonna di zeri aggiunta a destra di B produce una colonna di
#     zeri a destra di C, anch'essa da rimuovere;
#   - una colonna di zeri aggiunta ad A deve essere accompagnata da una
#     riga di zeri in B (le due dimensioni devono restare compatibili), e
#     il loro contributo al prodotto e' nullo: C non ne risente affatto.
#
# Il prezzo da pagare e' che il controllo e l'eventuale copia delle
# matrici avvengono ad ogni livello della ricorsione.

# Funzione ricorsiva che esegue l'algoritmo di Strassen con l'approccio
# dynamic padding.
# Accetta matrici di qualsiasi dimensione poichè ad ogni chiamata
# aggiunge una riga/colonna di zeri nel caso in cui una delle
# dimensioni sia dispari.
## Argomenti:
## @A: la prima matrice della moltiplicazione
## @B: la seconda matrice della moltiplicazione
## @Q: il cross-over point
### Return:
### @C: la matrice risultante
def Strassen_dynamic_padding(A,B,Q):

    RA, CA = A.shape #variabile che contiene le dimensioni della prima matrice
    RB, CB = B.shape #variabile che contiene le dimensioni della seconda matrice

    # Se le matrici sono più piccole del "crossover point" (Q)
    # il criterio adottato qui e' la media geometrica delle tre dimensioni;
    # il secondo controllo evita di partizionare matrici cosi' piccole da
    # rendere la ricorsione inutile
    if (RA*RB*CB)**(1/3) <= Q or min(RA,CA,CB) <= 2:
        # eseguo il classico algoritmo riga per colonna
        return nmp.matmul(A,B)

    # Se le matrici NON sono più piccole del "crossover point" (Q)

    drop_c = False #booleano che diventa True se aggiungo almeno una colonna
    drop_r = False #booleano che diventa True se aggiungo almeno una riga

    if RA%2: #se il numero di righe è dispari
        #nmp.pad con ((0,1),(0,0)) aggiunge una riga di zeri in fondo
        A = nmp.pad(A, ((0,1), (0,0)) )
        drop_r = True #booleano che indica che ho aggiunto una riga
        RA += 1 #variabile che contiene il numero di righe della prima matrice

    if CA%2: #se il numero di colonne è dispari

        #qui servono due interventi coordinati: una colonna di zeri a destra
        #di A e la corrispondente riga di zeri in fondo a B, per mantenere
        #compatibili le dimensioni interne del prodotto
        A = nmp.pad(A, ((0,0), (0,1)) )
        CA += 1 #aggiorno la variabile con le dimensioni di A
        B = nmp.pad(B, ((0,1), (0,0)) )
        RB += 1 #aggiorno anche il numero di righe di B

    if CB%2: #se il numero di colonne è dispari
        #colonna di zeri a destra di B: andra' poi rimossa da C
        B = nmp.pad(B, ((0,0), (0,1)) )
        drop_c = True
        CB += 1

    # In seguito verrà partizionata la matrice A in blocchi in maniera
    # analoga a quanto fatto da Strassen nel suo paper
    # (a questo punto tutte le dimensioni sono pari, quindi i quattro
    # blocchi hanno sicuramente le stesse misure)
    A11, A12, A21, A22 = split_matrix(A)
    B11, B12, B21, B22 = split_matrix(B)

    # eseguo i prodotti indicati da Strassen in maniera ricorsiva
    # usando questa stessa funzione ("Strassen_dynamic_padding")
    M1 = Strassen_dynamic_padding(A11+A22, B11+B22,Q)
    M2 = Strassen_dynamic_padding(A21+A22, B11,Q)
    M3 = Strassen_dynamic_padding(A11, B12-B22,Q)
    M4 = Strassen_dynamic_padding(A22, B21-B11,Q)
    M5 = Strassen_dynamic_padding(A11+A12, B22,Q)
    M6 = Strassen_dynamic_padding(A21-A11, B11+B12,Q)
    M7 = Strassen_dynamic_padding(A12-A22, B21+B22,Q)

    # sommmo le matrici come indicato da Strassen
    # per trovare i blocchi della matrice finale
    C11, C12, C21, C22 = strassen_sums(M1, M2, M3, M4, M5, M6, M7)

    # unisco i blocchi della matrice finale per
    # trovare la matrice risultante
    C = merge_matrix(C11, C12, C21, C22)

    #controllo se è necessario rimuovere eventuali
    #righe/colonne di zeri
    #ossia quelle introdotte in questa stessa chiamata, segnalate dai due
    #booleani: il risultato restituito al chiamante ha quindi sempre le
    #dimensioni "vere", come se il padding non fosse mai avvenuto
    if drop_r:
        C = nmp.delete(C,RA-1,0) #togli l'ultima riga
    if drop_c:
        C = nmp.delete(C,CB-1,1) #togli l'ultima colonna

    return C

#############################################################################
######################### ALGORITMO "STATIC PADDING" ########################
#############################################################################

# Secondo approccio: tutto il padding viene deciso e applicato una sola
# volta, prima di iniziare la ricorsione. L'idea e' calcolare quante
# righe/colonne di zeri servono perche' nessuna dimensione diventi dispari
# lungo tutta la discesa ricorsiva, fino al crossover point.
#
# Come si ricava quel numero:
#   1. si conta z, il numero di divisioni per due necessarie per portare la
#      dimensione minima sotto Q (ossia la profondita' della ricorsione);
#   2. una dimensione resta pari per z divisioni consecutive se e solo se e'
#      multipla di 2^z;
#   3. si arrotonda quindi ogni dimensione al piu' piccolo multiplo di 2^z
#      maggiore o uguale ad essa.
#
# Il vantaggio e' che i controlli sulle dimensioni e le copie di memoria
# avvengono una volta sola, non ad ogni livello; lo svantaggio e' che si
# possono aggiungere piu' zeri del minimo indispensabile.
# Il prodotto vero e proprio e' delegato a Strassen_rectM().

# Funzione che restituisce "True" se il valore è una potenza di 2.
## Argomenti:
## @n il numero di interesse.
### Return:
### @booleano, "True" se è potenza di 2, no altrimenti.
#
# Il logaritmo in base 2 di una potenza di 2 e' intero, quindi ha parte
# decimale nulla: il resto della divisione per 1 e' lo scarto decimale.
def is_power_of_two(n:int) -> bool:
    return math.log(n,2)%1==0

# Funzione per verificare se una dimensione è pari fino al
# raggiungimento del crossover point.
## Argomenti:
## @n, il numero di righe/colonne
## @Q, il cross-over point
### Return:
### @booleano, "True" rimane sempre pari, "False" altrimenti.
#
# Simula la discesa ricorsiva: finche' la dimensione supera Q la dimezza e
# verifica che resti pari ad ogni passo. Il primo valore dispari incontrato
# e' quello che bloccherebbe il partizionamento in blocchi.
def is_even_recursively(n:int, Q:int) -> bool:
    while n > Q:
        if n % 2 == 1:
            return False
        n = n / 2
    return True

# Funzione per calcolare il numero di chiamate ricorsive
# per raggiungere il crossover point.
## Argomenti:
## @n, il numero di righe/colonne
## @Q, il cross-over point
### Return:
### @int, il numero di chiamate ricorsive necessarie.
#
# Dimezzare n per z volte lo porta a n/2^z: imporre n/2^z <= Q significa
# z >= log2(n/Q), da cui l'arrotondamento per eccesso.
def n_recursion_needed(n:int,Q:int):
    x = math.log(n/Q,2)
    ris = math.ceil(x)
    return ris

# Funzione per determinare il numero minimo di righe/colonne
# che una matrice deve avere per poter rimanere a dimensioni pari
# fino al raggiungimento del crossover point
## Argomenti:
## @n, il numero di righe/colonne
## @n_rec, il numero di ricorsioni necessarie per superare il crossover point
### Return:
### @int, il numero di righe/colonne necessarie
#
# Restituisce il piu' piccolo multiplo di 2^n_rec maggiore o uguale alla
# dimensione data: dividendo per 2^n_rec, arrotondando per eccesso e
# rimoltiplicando si ottiene esattamente quel valore.
def n_row_or_col_needed(RMorCM, n_rec):
    return math.ceil(RMorCM/2**n_rec)*2**n_rec

# Funzione che esegue l'algoritmo di Strassen
# chiamando Strassen_naive(A,B,Q) [vedi sopra].
# Accetta matrici di qualsiasi dimensione poichè
# alla prima chiamata le aggiunge abbastanza
# righe/colonne di 0 in modo tale che rimangano di
# dimensioni pari ad ogni chiamata ricorsiva
## Argomenti:
## @A: la prima matrice della moltiplicazione
## @B: la seconda matrice della moltiplicazione
## @Q: il cross-over point
## Return:
## @C: la matrice risultante
def Strassen_static_padding(A,B,Q:int):

    rows_added = 0 #intero che indica le colonne aggiunte
    cols_added = 0 #intero che indica le righe aggiunte

    RA, CA = A.shape #variabile che contiene le dimensioni della prima matrice
    RB, CB = B.shape #variabile che contiene le dimensioni della seconda matrice

    # calcolo del numero di ricorsioni necessarie fino al raggiungimento del cross-over point
    # si parte dalla dimensione minore, coerentemente con il criterio di
    # arresto di Strassen_rectM(), che si ferma appena una delle tre
    # dimensioni scende sotto Q
    z = n_recursion_needed(min(RA, CA, RB, CB), Q)

    # aggiunge il numero di righe a A in base all'esegenza
    # rows_added viene memorizzato perche' a tante righe di zeri in A
    # corrispondono altrettante righe di zeri da togliere in fondo a C
    if not is_even_recursively(RA,Q):
        x = n_row_or_col_needed(RA, z)
        rows_added = x-RA
        A = nmp.pad(A, ((0,rows_added), (0,0)) )
        RA = x

    # aggiunge il numero di colonne ad A e di righe a B in base all'esigenza
    # questa dimensione e' condivisa (le colonne di A sono le righe di B):
    # va quindi estesa in entrambe le matrici, e non influenza C
    if not is_even_recursively(CA,Q):
        x = n_row_or_col_needed(CA, z)
        A = nmp.pad(A, ((0,0), (0,x-CA)) )
        B = nmp.pad(B, ((0,x-RB), (0,0)) )
        CA = RB = x

    # aggiunge il numero di colonne a B in base all'esegenza
    # cols_added corrisponde alle colonne di zeri da togliere a destra di C
    if not is_even_recursively(CB,Q):
        x = n_row_or_col_needed(CB, z)
        cols_added = x-CB
        B = nmp.pad(B, ((0,0), (0,cols_added)) )
        CB = x

    C = Strassen_rectM(A,B,Q) #chiamata a Strassen_naive che risolve ricorsivamente
                              #il problema
                              #a questo punto le dimensioni sono garantite pari
                              #per tutta la discesa, quindi la variante
                              #rettangolare "pura" e' sufficiente

    #controllo se è necessario rimuovere eventuali righe di zeri
    #nmp.delete riceve la lista degli indici delle ultime rows_added righe
    if rows_added > 0:
        C = nmp.delete(C,list(range(RA-rows_added, RA,1)),0) #togle le righe extra

    #controllo se è necessario rimuovere eventuali colonne di zeri
    #stessa logica sulle colonne, con l'asse 1 invece dell'asse 0
    if cols_added > 0:
        C = nmp.delete(C,list(range(CB-cols_added, CB,1)),1) #togle le colonne extra

    return C

#############################################################################
######################## ALGORITMO "DYNAMIC PEELING" ########################
#############################################################################

# Terzo approccio: invece di aggiungere zeri si togliono righe e colonne.
# Se una dimensione e' dispari si mette da parte l'ultima riga di A e/o
# l'ultima colonna di B, si applica Strassen alla sottomatrice rimasta
# (che ha dimensioni pari) e si recuperano i contributi mancanti con il
# prodotto riga per colonna, secondo la teoria delle matrici a blocchi.
#
# Nel caso in cui tutte e tre le dimensioni siano dispari, partizionando
#
#       A = | A11  a12 |        B = | B11  b12 |
#           | a21  a22 |            | b21  b22 |
#
# dove A11 e B11 sono le matrici private di ultima riga e ultima colonna,
# a12 e b12 sono vettori colonna, a21 e b21 vettori riga e a22, b22 scalari,
# si ottiene
#
#       C = | A11*B11 + a12*b21    A11*b12 + a12*b22 |
#           | a21*B11 + a22*b21    a21*b12 + a22*b22 |
#
# Di questi prodotti solo A11*B11 e' abbastanza grande da valere una
# chiamata ricorsiva: in tutti gli altri compare un vettore o uno scalare,
# quindi si risolvono direttamente con numpy.matmul().
#
# La funzione distingue esplicitamente le 8 combinazioni possibili di
# parita' delle tre dimensioni (2^3 = 8 casi), ciascuna con il proprio
# schema a blocchi; l'ultimo caso, tutte le dimensioni pari, e' il passo
# di Strassen classico.
#
# Riferimento: "Implementation of Strassen's Algorithm for Matrix
# Multiplication", Huss-Lederman, Jacobson, Johnson, Tsao, Turnbull (1996).

# Funzione per dividere una matrice in due blocchi, di cui uno
# è l'ultima riga
## Argomenti:
## @M, la matrice
### Return:
### @M11, @m21 ossia i 2 blocchi
def split_last_r(M):
    RM, CM = M.shape
    M11 = M[0:(RM-1),:]  #tutta la matrice tranne l'ultima riga
    m21 = M[(RM-1),:]    #solo l'ultima riga
    m21.shape = (1,CM) #fissa le dimensioni per evitare problemi
                            #nei calcoli matriciali successivi
                            #(lo slicing di una singola riga restituisce un
                            #array monodimensionale, qui serve una matrice
                            #[1 x CM])
    return M11, m21

# Funzione per dividere una matrice in due blocchi, di cui uno
# è l'ultima colonna
## Argomenti:
## @M, la matrice
### Return:
### @M11, @m12 ossia i 2 blocchi
def split_last_c(M):
    RM, CM = M.shape
    M11 = M[:,0:(CM-1)]  #tutta la matrice tranne l'ultima colonna
    m12 = M[:,(CM-1)]    #solo l'ultima colonna
    m12.shape = (RM,1) #fissa le dimensioni per evitare problemi
                            #nei calcoli matriciali successivi
                            #(da array monodimensionale a matrice [RM x 1])
    return M11, m12

# Funzione per dividere una matrice M[j,k] in quattro blocchi, di cui uno
# di dimensioni [1,k-1], uno di dimensioni [j-1,1] e uno di dimensioni [1,1]
## Argomenti:
## @M, la matrice
### Return:
### @M11, @m12, @m21, @m22 ossia i 4 blocchi
#
# E' il partizionamento usato quando sia le righe sia le colonne sono in
# numero dispari: si isolano contemporaneamente l'ultima riga, l'ultima
# colonna e l'elemento che si trova all'incrocio tra le due.
def split_last_rc(M):
    RM, CM = M.shape
    M11 = M[0:(RM-1),0:(CM-1)]              #blocco principale, dimensioni pari
    m12 = nmp.matrix(M[0:(RM-1),(CM-1)])    #ultima colonna, escluso l'angolo
    m12.shape = (RM-1,1) #fissa le dimensioni per evitare problemi
                            #nei calcoli matriciali successivi
    m21 = nmp.matrix(M[RM-1,0:(CM-1)])      #ultima riga, escluso l'angolo
    m21.shape = (1,CM-1) #fissa le dimensioni per evitare problemi
                            #nei calcoli matriciali successivi
    m22 = nmp.matrix(M[(RM-1), (CM-1)])     #l'elemento nell'angolo in basso a destra
    m22.shape = (1,1) #fissa le dimensioni per evitare problemi
                            #nei calcoli matriciali successivi
    return M11, m12, m21, m22

# Funzione che esegue l'algoritmo di Strassen nella versione
# "dynamic peeling". Accetta matrici di qualsiasi dimensione poichè
# ad ogni chiamata toglie una riga/colonna in caso una di queste dimensioni
# sia dispari. Il cross-over point è dato dalla media geometrica delle dimensioni.
## Argomenti:
## @A: la prima matrice della moltiplicazione
## @B: la seconda matrice della moltiplicazione
## @Q: il cross-over point
## Return:
## @C: la matrice risultante
def Strassen_dynamic_peeling(A,B,Q:int):

    RA, CA = A.shape #variabile che contiene le dimensioni della prima matrice
    RB, CB = B.shape #variabile che contiene le dimensioni della seconda matrice

    #se la media geometrica delle dimensioni è più piccola del cross-over point
    # o almeno una delle dimensioni è minore o uguali a 2
    #(stesso criterio del dynamic padding: caso base della ricorsione)
    if (RA*CA*CB)**(1/3) <= Q  or min(RA,CA,CB) <= 2:
        return nmp.matmul(A,B)

    a_odd = False #se il numero di righe di A è dispari
    b_odd = False #se il numero di colonne di A/numero di righe di B è dispari
    c_odd = False #se il numero di colonne di B è dispari

    if RA%2: #se il numero di righe di A è dispari
        a_odd = True

    if CA%2: #se il numero di colonne di A è dispari
    #il numero di righe di B è uguale al numero di colonne di A
        b_odd = True

    if CB%2: #se il numero di colonne di B è dispari
        c_odd = True

    # 8 combinazioni possibili:
    # i tre booleani appena calcolati individuano uno degli 8 casi che
    # seguono; per ciascuno si applica il corrispondente schema a blocchi

    # 1)
    # solo prima dimensione dispari
    # si stacca l'ultima riga di A: produce l'ultima riga di C
    if a_odd and (not b_odd) and (not c_odd):

        #svolgimento delle operazioni come indicate in:
        #"Implementation of Strassen’s Algorithm for Matrix Multiplication"
        A11, a21 = split_last_r(A)
        C11 = Strassen_dynamic_peeling(A11,B,Q)   #blocco grande, ricorsione

        C21 = nmp.matmul( a21, B)                 #riga per colonna sul vettore
        return nmp.vstack((C11,C21))              #i due blocchi si impilano

    # 2)
    # solo seconda dimensione dispari
    # la dimensione "interna" dispari non cambia la forma di C: il
    # contributo della colonna/riga staccata e' un prodotto esterno che si
    # somma al risultato
    elif (not a_odd) and b_odd and (not c_odd):

        #svolgimento delle operazioni come indicate in:
        #"Implementation of Strassen’s Algorithm for Matrix Multiplication"
        A11, a12 = split_last_c(A)
        B11, b21 = split_last_r(B)

        C = Strassen_dynamic_peeling(A11,B11,Q)
        C += nmp.matmul(a12, b21)   #prodotto esterno colonna x riga
        return C

    # 3)
    # solo terza dimensione dispari
    # si stacca l'ultima colonna di B: produce l'ultima colonna di C
    elif (not a_odd) and (not b_odd) and c_odd:

        #svolgimento delle operazioni come indicate in:
        #"Implementation of Strassen’s Algorithm for Matrix Multiplication"
        B11, b12 = split_last_c(B)

        C11 = Strassen_dynamic_peeling(A, B11, Q)
        C12 = nmp.matmul(A, b12)
        return nmp.hstack((C11,C12))   #i due blocchi si affiancano

    # 4)
    #solo prima e seconda dimensione dispari
    #A viene partizionata in quattro blocchi, B solo per righe
    elif a_odd and b_odd and (not c_odd):

        #svolgimento delle operazioni come indicate in:
        #"Implementation of Strassen’s Algorithm for Matrix Multiplication"
        A11, a12, a21, a22 = split_last_rc(A)
        B11, b21 = split_last_r(B)

        C11 = Strassen_dynamic_peeling( A11, B11, Q)
        C11 += nmp.matmul( a12, b21 )

        C21 = nmp.matmul( a21, B11 ) + nmp.matmul( a22, b21 )

        return nmp.vstack((C11,C21))

    # 5)
    #solo prima e terza dimensione dispari
    #e' l'unico caso in cui C si ricompone da quattro blocchi "pieni"
    elif a_odd and (not b_odd) and c_odd:

        #svolgimento delle operazioni come indicate in:
        #"Implementation of Strassen’s Algorithm for Matrix Multiplication"
        A11, a21 = split_last_r(A)
        B11, b12 = split_last_c(B)

        C11 = Strassen_dynamic_peeling(A11, B11, Q)
        C12 = nmp.matmul(A11, b12)
        C21 = nmp.matmul(a21, B11)
        C22 = nmp.matmul(a21, b12 )   #scalare: incrocio di riga e colonna

        return merge_matrix(C11,C12,C21,C22)

    # 6)
    #solo seconda e terza dimensione dispari
    #simmetrico del caso 4: qui e' B a essere partizionata in quattro blocchi
    elif (not a_odd) and b_odd and c_odd:

        #svolgimento delle operazioni come indicate in:
        #"Implementation of Strassen’s Algorithm for Matrix Multiplication"
        A11, a12 = split_last_c(A)
        B11, b12, b21, b22 = split_last_rc(B)

        C11 = Strassen_dynamic_peeling( A11, B11, Q)
        C11 += nmp.matmul( a12, b21 )

        C21 = nmp.matmul( A11, b12 ) + nmp.matmul( a12, b22 )

        return nmp.hstack((C11,C21))

    # 7)
    #tutte le dimensioni dispari
    #e' il caso generale descritto nell'intestazione della sezione: una
    #sola chiamata ricorsiva (A11*B11) e sette prodotti con vettori o scalari
    elif a_odd and b_odd and c_odd:

        #svolgimento delle operazioni come indicate in:
        #"Implementation of Strassen’s Algorithm for Matrix Multiplication"
        A11, a12, a21, a22 = split_last_rc(A)
        B11, b12, b21, b22 = split_last_rc(B)

        C11 = Strassen_dynamic_peeling( A11, B11, Q)
        C11 += nmp.matmul( a12, b21 )

        C12 = nmp.matmul(A11, b12) + nmp.matmul(a12, b22)

        C21 = nmp.matmul( a21, B11 ) + nmp.matmul( a22, b21 )

        C22 = nmp.matmul(a21,b12) + nmp.matmul(a22, b22)

        return merge_matrix(C11,C12,C21,C22)

    # 8:
    # tutte le dimensioni sono pari
    # nessuna riga o colonna da sbucciare: si procede con il passo di
    # Strassen classico, identico a quello delle altre varianti
    elif (not a_odd) and (not b_odd) and (not c_odd):

        #svolgimento delle operazioni come indicate in:
        #"Implementation of Strassen’s Algorithm for Matrix Multiplication"
        A11, A12, A21, A22 = split_matrix(A)
        B11, B12, B21, B22 = split_matrix(B)

        M1 = Strassen_dynamic_peeling(A11+A22, B11+B22,Q)
        M2 = Strassen_dynamic_peeling(A21+A22, B11,Q)
        M3 = Strassen_dynamic_peeling(A11, B12-B22,Q)
        M4 = Strassen_dynamic_peeling(A22, B21-B11,Q)
        M5 = Strassen_dynamic_peeling(A11+A12, B22,Q)
        M6 = Strassen_dynamic_peeling(A21-A11, B11+B12,Q)
        M7 = Strassen_dynamic_peeling(A12-A22, B21+B22,Q)

        C11, C12, C21, C22 = strassen_sums(M1, M2, M3, M4, M5, M6, M7)

        C = merge_matrix(C11, C12, C21, C22)

        return C

    #se non rientra in nessuno dei casi precedenti
    #gli 8 rami coprono tutte le combinazioni, quindi questa riga funge da
    #rete di sicurezza
    raise ValueError("Errore nel dynamic peeling")
