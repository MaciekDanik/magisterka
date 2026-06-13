# Raport naprawczy logiki i stylu pracy

## Zakres audytu

Sprawdzono spójność logiczną rozdziałów, zgodność opisów z artefaktami repozytorium oraz styl akademicki pracy magisterskiej. Punktem odniesienia były pliki `rozdzialy/*.tex`, tabela wyników, konfiguracje zbiorów danych, katalogi `datasets`, katalogi `models/training_runs` oraz skrypty w `results/scripts`.

## Ustalenia i działania

1. **Kryterium wyboru przebiegu treningowego**  
   W skryptach wynikowych celowo wybierany jest najdłuższy przebieg treningowy. Pozostawiono to bez zmiany, ponieważ krótsze przebiegi pełnią funkcję kontrolną/testową. W rozdziale wynikowym doprecyzowano, że do zestawień użyto najdłuższych kompletnych przebiegów.

2. **Rozróżnienie Metody A i Metody B**  
   Metoda A jest uczeniem półnadzorowanym z pseudoetykietowaniem, natomiast Metoda B jest syntetyczną augmentacją próbkami negatywnymi. Poprawiono miejsca, w których obie metody były zbiorczo opisywane jako SSL lub w których syntetyczne tła nazwano danymi nieoznakowanymi.

3. **Liczebność danych dodatkowych**  
   Repozytorium zawiera 60 obrazów/pseudoetykiet w `datasets/pseudo` oraz 89 syntetycznych próbek tła w `datasets/generative_negatives`. Skorygowano opis kierunków dalszych badań, gdzie wcześniej liczby te były ujęte nieprecyzyjnie.

4. **Komentarze robocze w rozdziale wynikowym**  
   Usunięto komentarze typu `% MIEJSCE NA ...`, ponieważ nie powinny pozostać w finalnym źródle pracy.

5. **Stylistyka akademicka**  
   Ograniczono zbyt mocne lub czasowo nietrwałe sformułowania, m.in. „najnowsza”, „drastycznie”, „ogromne”, „wysoce”, tam gdzie nie były niezbędne merytorycznie.

6. **Spójność wykresów i podpisów**  
   Ujednolicono nazewnictwo w skryptach wykresów: „Model odniesienia”, „Metoda A (pseudoetykiety)” oraz „Metoda B (syntetyczne tła)”. Zmieniono tytuły wykresów z „metod SSL” na neutralne „metody wzbogacania danych”.

## Status

Raport został zrealizowany w zmianach tekstowych oraz w skryptach generujących wykresy. Wykresy zostały odświeżone i przeniesione do katalogu `M_Danik_Praca_magisterska/Pics/Charts`.
