Tanken är att i detta projekt tillverka en "robot" som spelar det traditionella spelet labyrint.

Spelet går till så att en kula läggs på en platta med en labyrint i. Labyrinten har flera fällor/hål
där kulan kan ramla ner. man kan luta plattan genom att vrida på två stycken rattar.
när plattan lutar kommer kulan att börja rulla och på detta sätt skall man försöka ta sig igenom labyrinten

Steg i processen

1. Bygga roboten
	Tanken är att en enkel arduino och två servon skall agera output.
	Servon monteras på labyrintens rattar för att kunna luta plattan.

2. Programmera Arduino
	Arduinon kommer att programmeras utan någon logik inbyggd.
	Den kommer att lyssna på terminalen efter kommandon i stil med:
	Servo1: 2 #Vilket kommer vrida servo nummer 1 2 grader

3. Kamera för övervakning.
	En raspberry pi zero som är programmerad till att fungera som en övervakningskamera.
	Hårdvara: Raspberry pi Zero W
	Raspberry pi HQ camera
	Denna delen är redan färdig sedan ett tidigare projekt.

4. Python script för inlärning.
	En dator med med linux kommer kommunicera med arduinon via USB sladd och skicka kommandon.
	Denna datorn kommer att ta bilder från kameran med hjälp av opencv.
	Bilderna kommer att skalas ner och justeras för att isolera de objekt som är vesäntliga för koden.

5. Ev modifiera Labyrint spelet.
	Då kameran eventuellt kommer ha svårt att urskilja väggarna på plattan och hålen i brädet så kommer jag
	förmodligen behöver måla/tejpa vissa objekt så dom sticker ut färgmässigt.
	två ytterliggare servon kommer monteras på labyrintspelet för att lyfta upp kula och återplacera den i det fall
	kulan faller ner i ett hål.
	
