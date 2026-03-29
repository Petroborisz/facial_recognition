# Arcfelismerő Rendszer 

Ebben a projektben egy arcfelismerő (klasszifikációs) neurális hálózat teljes fejlesztési folyamatán mentem végig a [LFW (Labeled Faces in the Wild) deep-funneled](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset) adatbázisán. A cél egy olyan modell felépítése és betanítása volt, amely képes azonosítani az adatbázisban szereplő személyeket a róluk készült képek alapján.

## A projekt felépítése 

Rövid összefoglaló a projekt főbb lépéseiről, melyeket el kellett végeznem annak során:

1. **Adathalmaz feltérképezése:** A probléma definíciója és az UMASS LFW deep-funneled (előzetesen igazított) adatbázisának bemutatása. Ebben a részben látható a képek eloszlása, a címkék száma, és az adatbázisra jellemző extrém kiegyensúlyozatlanság.
2. **Adatok előfeldolgozása (Preprocessing):** A képek betöltése, normalizálása és a megfelelő tenzor-formátumok kialakítása a neurális hálózat számára. 
3. **Modell architektúra építése:** A hálózat gerincét (backbone) egy előre betanított **MobileNetV2** adja. Ezt a projekt során egy teljesen saját fejlesztésű, a sokosztályos problémára optimalizált klasszifikációs fejjel (custom head) egészítettem ki.
4. **Tanítás és Finomhangolás (Fine-Tuning):** A transzfer tanulás (transfer learning) gyakorlati megvalósítása. Itt próbáltan ki a finomhangolás minden stratégiáját, ami csak eszembe jutott. Ezek között főként a befagyasztott layerekkel való játék szerepel, hiszen ez bizonyosult a leghasznosabbnak. Emellett augmentációt alkalmaztam, melyben szerepelt a képek elforgatása, skálázása, ezáltal végezve aktívabb zajszűrést. 
5. **Eredmények kiértékelése:** A veszteség (loss) és a pontosság (accuracy) alakulásának elemzése a tanítási és validációs adatokon.
6. **Tesztelés és vizualizáció:** Konkrét predikciók lefuttatása és megjelenítése tesztképeken, hogy kicsit gyakorlatiasabban, játékosabban és végre vizuálisan is ellenőrizzem a modell döntéseit.

## Eredmények és Konklúzió

A legjobb modell a tesztelés során megközelítőleg **14%-os pontosságot (accuracy)** ért el. 

Ez az érték alacsonynak tűnhet a bináris klasszifikációs feladatoknál megszokott 90%+ os eredményekhez képest, azonban a probléma kontextusában ez mégis jó eredmény:
* **Hatalmas címketér (Label Space):** több mint 5000 különböző osztály (személy) között kell különbséget tennie a modellnek.
* **Extrém kiegyensúlyozatlanság:** Az adatbázisban szereplő személyek jelentős részéről mindössze 1-2 kép áll rendelkezésre, ami drasztikusan megnehezíti a stabil mintázatok megtanulását a hagyományos softmax klasszifikáció számára.
* **Véletlenszerű tippelés aránya:** Egy ilyen sokosztályos térben a véletlenszerű tippelés pontossága a 0.01%-ot sem érné el. A 14%-os pontosság azt bizonyítja, hogy a modell sikeresen megtanulta az arcok egyedi vonásait (feature extraction), és komoly prediktív erővel rendelkezik.

## Tapasztalt nehézségek
* **erőforrások**: A legnagyobb gát a projekt során az erőforrások hiányának bizonyult. Tisztában voltam a tanítás időigényességével, azonban sosem gondoltam volna, hogy ennek mértéke ilyen "egyszerű" tanítás során is ekkorára nőhet. Szintén nem sejtettem, hogy érdemesebb lesz beruházni is gyorsabb kernelre, amit végül nem tettem. A gooogle T4 GPU-ját használva az első nagyobb tanítás után elfogyott az összes ingyenesen használható futtatási egyságem. A tanítások nagyrészét a saját laptopomnak kellett elvégeznie. Emiatt esetenként másfél órás tanítások zajlottak, amikenk megszakítatlanul kellett történnie. 
* **fájlkezelés**: A kód használ egy kinevezett mappát jelenleg a legjobb modell elmentésére, azonban nem csináltam külön minden modell elmentésére mappát. Mindig a legutolsót menti bele, függetlenül attól, hogy jobb-e. (Azért legjobb modell, mert az adott tanításban az újabb és újabb epoch-ok közül a legjobbat tartalmazza) Ennek eredményeképp a tényleges legjobb modellt újra kell futtatni, ami az előző pontban leírt kernel-probléma miatt időigényes.

## Továbbfejlesztési lehetőségek

A projekt kiváló alapot nyújt a jövőbeli optimalizációhoz. A teljesítmény további növelése érdekében az alábbi irányokba érdemes elindulni:
* **További augmentáció**: mivel ezt a vonalat kevesebb fajta tanítással, kombinációkkal próbáltam ki, rengeteg fejlesztési lehetőség lakozik benne.
* **Hibás tesztelés**: az augmentáció utáni tesztelésnél jelenleg a modell augmentált képeken végzi a tesztet ami "csalás". Ez a projekt jövőbeli folytatásánál kezdőpont a fejlesztésben, hiszen itt mondhatni hibás a logika.
* **Alternatív Architektúrák:** A MobileNetV2 kiváló egyensúlyt ad a sebesség és pontosság között, de számításigényesebb backbone-ok (pl. ResNet50, EfficientNet) használata magasabb pontosságot eredményezhet.

## Használat / Futtatás
A projekthez tartozó kód és a teljes elemzés megtalálható a `facial_recognition.ipynb` fájlban. A futtatáshoz szükséges képfájlok a leírás elején található linken letölthető .zip fájlként, amelyben személyenként rendszerezve megtalálhatóak a képfájlok. A .ipynb file letöltésével a futtatott cellák eredményei megtekinthetőek. A saját vizuális teszteléshez szükséges futtatni az összes addigi cellát sorrendben.
