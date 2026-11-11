# Questions the final agent (leaderboard 0.6234) gets wrong or only partly right

Source: full server run of the submitted agent (runs/FINAL_full7). F1 is answer F1 against the gold answer and its aliases.

## Summary

| Reason | Questions | F1 lost |
|---|---|---|
| Wrong bridge | 49 | 44.0 |
| Bridge right, but the answer passage was never retrieved | 31 | 25.7 |
| Bridge right, answer passage shown, but the model read the wrong fact | 29 | 29.0 |
| Partly right | 15 | 6.0 |
| No bridge | 8 | 8.0 |
| Bridge right, answer passage retrieved but never shown to the model | 3 | 3.0 |
| **Total** | **135** | **115.6** |

## All questions

| # | Question | Gold answer | Our answer | F1 | Our bridge | Gold bridge | Why it fails |
|---|---|---|---|---|---|---|---|
| 1 | When is the last time auburn won in the work location of Johnny Berthelot? | 1999 | 1994 | 0.00 | Gonzales | Baton Rouge | Wrong bridge: right passage was retrieved but the model picked another entity |
| 3 | Which region is the city served by CIWW located? | Franklin County | Ottawa | 0.00 | Ottawa | Ottawa | Bridge right, answer passage shown, but the model read the wrong fact |
| 4 | What is the population of Arlene B. Arcillas's birthplace in California? | 175,155 | (empty) | 0.00 | (none) | Santa Rosa | No bridge: hop 1 found nothing usable |
| 6 | Where did the team that won the MLB World Series in 2007 get Brandon Phillips from? | Los Angeles Angels | Cleveland Indians | 0.00 | Boston Red Sox | the Red Sox | Bridge right, answer passage shown, but the model read the wrong fact |
| 9 | Where is the rainforest in the country where Aníbal Acevedo Vilá has citizenship located? | slopes of the Sierra de Luquillo mountains | United States | 0.00 | United States | Puerto Rico | Wrong bridge: right passage was retrieved but the model picked another entity |
| 10 | Dean, in the country where Trailer Park Boys takes place, shares a border with what county? | Colchester County | Whatcom County | 0.50 | Canada | Nova Scotia | Wrong bridge: right passage was retrieved but the model picked another entity |
| 11 | What musical ensemble released a live album recorded at the broadcasting studios of the network that distributed Vision On? | Fleetwood Mac | IAMX | 0.00 | Vision Film | BBC | Wrong bridge: the hop-1 passage was never retrieved |
| 17 | What was the former name of the birthplace of Lou Yun? | Qiantang | Hangzhou | 0.00 | (none) | Hangzhou | No bridge: hop 1 found nothing usable |
| 18 | When was the capitol of the state with a buffalo on its license plate built? | between 1931 and 1934 | 1931 | 0.40 | North Dakota | North Dakota | Partly right: answer wording/span differs from gold |
| 21 | The number of Imax screens in the US and Canada showing the film with a wider UK release than Skyfall was how many? | 374 | 388 | 0.00 | 40 | Spectre. | Wrong bridge: right passage was retrieved but the model picked another entity |
| 25 | Who is part of the band that performed the song English Settlement? | Andy Partridge | XTC | 0.00 | XTC | XTC | Bridge right, answer passage shown, but the model read the wrong fact |
| 26 | Who is the sibling of the composer of Downtown? | Astrid Young | Michael Gore | 0.00 | Luke Laird | Neil Young | Wrong bridge: the hop-1 passage was never retrieved |
| 29 | Who was targeted for defrauding the electoral process, in the state Hotel Courtland is located? | the Republicans are willing to do things that are unethical to manipulate elections | John Kerry | 0.00 | Ohio | Ohio | Bridge right, but the answer passage was never retrieved |
| 31 | How old do you have to be to drink with a parent in the state where WJQM is located? | 18 - 20 | 21 | 0.00 | Wisconsin | Wisconsin | Bridge right, answer passage shown, but the model read the wrong fact |
| 33 | What forest is by the southern border of the death city of Jerzy Passendorfer? | Kabaty | Kabaty Forest | 0.67 | Skolimów | Warsaw | Wrong bridge: right passage was retrieved but the model picked another entity |
| 34 | When did the river on which the Palo Verde Dam is located start forming the Grand Canyon? | about 5 to 6 million years ago | 5 to 6 million years ago | 0.92 | Colorado River | Colorado River | Partly right: answer wording/span differs from gold |
| 37 | The creator of Watching the Wheels is whose son? | Alfred Lennon | Julian Lennon | 0.50 | John Lennon | John Lennon | Partly right: answer wording/span differs from gold |
| 39 | What is the record label of the performer of Father and Daughter? | Warner Bros. | Columbia | 0.00 | Paul Simon | Paul Simon | Bridge right, but the answer passage was never retrieved |
| 40 | When did the US go to war with the country where Ofusato died? | the Battle of Osan | 1950 | 0.00 | Korea | Korea | Bridge right, answer passage shown, but the model read the wrong fact |
| 44 | What is the birth county of Norman Levinson? | Essex County | Hardin County | 0.50 | Lynn | Lynn | Bridge right, but the answer passage was never retrieved |
| 47 | What was the average household size in the city served by WAJM? | 2.46 | 2.50 | 0.00 | Atlantic City | Atlantic City | Bridge right, answer passage shown, but the model read the wrong fact |
| 48 | Which range is Garfield Peak in the state of Hinsdale County part of? | Sawatch Range | Collegiate Peaks | 0.00 | Sawatch Range | Colorado | Wrong bridge: right passage was retrieved but the model picked another entity |
| 51 | What is an annual event hosted by the largest metro area in Oklahoma as of 2010? | Big 12 Baseball Tournament | NCAA Women's College World Series | 0.00 | Oklahoma City | Oklahoma City | Bridge right, answer passage shown, but the model read the wrong fact |
| 52 | The album named after the lead guitar player on While My Guitar gently Weeps was produced by whom? | Eric Clapton | George Martin | 0.00 | Eric Clapton | Clapton | Bridge right, answer passage shown, but the model read the wrong fact |
| 53 | Whe did sales of petrol by liters begin in the country owning RAF Brize Norton? | 1976 | United Kingdom | 0.00 | United Kingdom | the UK | Wrong bridge: right passage was retrieved but the model picked another entity |
| 55 | who was manager when the team Tommy Logan plays for won champions league? | Roberto Di Matteo | José Mourinho | 0.00 | Chelsea | Chelsea | Bridge right, but the answer passage was never retrieved |
| 56 | Which language is spoken by the people that lived in the British Isles during the regime under which the oldest recorded incident of civil disobedience occurred? | Celtic dialects of the Insular Celtic group | Latin | 0.00 | Thebes | during the Roman Empire | Wrong bridge: right passage was retrieved but the model picked another entity |
| 59 | In WWII when did Australian troops fight in the birthplace of Erin Stewart? | October 1944 | May 4, 1987 | 0.00 | May 4, 1987 | New Britain | Wrong bridge: right passage was retrieved but the model picked another entity |
| 60 | How many members are in the agency from which countries are excluded due to lack of strategic and hard power? | fifteen | seven | 0.00 | seven | UN Security Council | Wrong bridge: right passage was retrieved but the model picked another entity |
| 61 | When was the brand opened in city where Auguste de Montferrand died? | 1855 | 1739 | 0.00 | St Petersburg | St Petersburg | Bridge right, but the answer passage was never retrieved |
| 62 | Who collaborated with the performer of Hero of the Day? | San Francisco Symphony | Lou Reed | 0.00 | Metallica | Metallica | Bridge right, but the answer passage was never retrieved |
| 66 | What is the highest point which it can be found in the country Galilaya is located? | 1,400 metres | Not available | 0.00 | Uganda | Uganda | Bridge right, but the answer passage was never retrieved |
| 67 | When were the mosaics at the church in the headquarter of IONOS created? | 5th–6th centuries | (empty) | 0.00 | (none) | Thessaloniki | No bridge: hop 1 found nothing usable (hop-1 passage never retrieved) |
| 68 | When was the first railway between Ali Abbasi's birth city and Kotri constructed? | April 1858 | 13 May 1861 | 0.00 | Karachi | Karachi | Bridge right, answer passage shown, but the model read the wrong fact |
| 69 | When did the civil war in the country Ita Ford is from start? | October 15, 1979 | April 1861 | 0.00 | United States | El Salvador | Wrong bridge: right passage was retrieved but the model picked another entity |
| 70 | Mark A. Carleton's birthplace shares a border with which other county? | Greenbrier County | Washington County | 0.50 | near Jerusalem, Monroe County, Ohio | Monroe County | Bridge right, but the answer passage was never retrieved |
| 71 | Who was in charge of the place where the Maratha Kingdom was located? | Prithviraj Chavan | Shivaji | 0.00 | western Deccan Plateau | Maharashtra | Wrong bridge: right passage was retrieved but the model picked another entity |
| 73 | Which art museum does the past employer of Catherine LaCugna administer? | Snite Museum of Art | Mr. Fezziwig | 0.00 | Mr. Fezziwig | Notre Dame | Wrong bridge: right passage was retrieved but the model picked another entity |
| 76 | What kind of agency did Dr. T.R.M. Howard criticize? | FBI is primarily a domestic agency | civil rights leaders | 0.00 | FBI | FBI | Bridge right, answer passage shown, but the model read the wrong fact |
| 79 | What tradition does the legal system, of the Chinese territory containing Coloane Park, come from? | Portuguese-based legal system | common law | 0.00 | Macau | Macau | Bridge right, answer passage shown, but the model read the wrong fact |
| 83 | When did the roof gardens above where Edwardes Square is located open to the public? | 1980s | (empty) | 0.00 | (none) | Kensington | No bridge: hop 1 found nothing usable |
| 84 | Where can you find the show ESPN College Football on the original broadcaster of Angie? | American Broadcasting Company | ABC | 0.00 | ABC | ABC | Bridge right, answer passage shown, but the model read the wrong fact |
| 88 | When was the last time Auburn won in Tim Breaux's birthplace? | 1999 | 2014 | 0.00 | Baton Rouge, Louisiana | Baton Rouge | Bridge right, answer passage shown, but the model read the wrong fact |
| 89 | Who is the performer of the album using the name of the country containing Urdaneta? | Aldemaro Romero | Jose Mari Chan | 0.00 | Philippines | Venezuela | Wrong bridge: right passage was retrieved but the model picked another entity |
| 90 | When did the 5th Dalai Lama gain political control over Marpa Lotsawa's place of birth? | In 1642 | 1642 | 0.67 | Guge | Tibet | Wrong bridge: right passage was retrieved but the model picked another entity |
| 92 | What is the name for the spirits associated with Shintoism, in the original language of the song Nobody Knows? | Kami | Spirits | 0.00 | Japanese | Japanese | Bridge right, but the answer passage was never retrieved |
| 94 | What was the first year men's basketball was played at Donald Worster's alma mater? | 1898 | 1921 - 22 | 0.00 | University of Kansas | University of Kansas | Bridge right, answer passage shown, but the model read the wrong fact |
| 95 | Why does the flag of the US territory that Luis Villafañe is from have one star? | represented the island of Puerto Rico | the island of Puerto Rico | 0.89 | Puerto Rico | Puerto Rico | Partly right: answer wording/span differs from gold |
| 100 | What conflict did the spouse of Maria Ley-Piscator fight in? | World War I | Spanish Civil War | 0.40 | Erwin Piscator | Erwin Piscator | Bridge right, but the answer passage was never retrieved |
| 103 | Who plays the creator of Sunflowers in Doctor Who? | Tony Curran | Leslie Megahey | 0.00 | Leslie Megahey | Vincent van Gogh | Wrong bridge: right passage was retrieved but the model picked another entity |
| 105 | Which is the body of water by the region containing Cary? | Irish Sea | north - Antrim coast | 0.00 | County Antrim | County Antrim | Bridge right, answer passage shown, but the model read the wrong fact |
| 107 | In what county can you find the city where the band To My Surprise formed? | Warren County | Polk County | 0.50 | Des Moines | Des Moines | Bridge right, but the answer passage was never retrieved |
| 108 | The city where Lakes International Language Academy is located is in which county? | Washington County | Becker County | 0.50 | Becker | Forest Lake | Wrong bridge: right passage was retrieved but the model picked another entity |
| 110 | What college did the performer of The Definitive Collection attend? | Syracuse University | University of Southern California | 0.33 | Michael Jackson | Lou Reed | Wrong bridge: right passage was retrieved but the model picked another entity |
| 112 | What county shares a border with the county of Ross Township? | Indiana County | Baranya County | 0.50 | Jefferson County | Jefferson County | Bridge right, but the answer passage was never retrieved |
| 118 | What county shares a border with another county that contains Pelton Dam? | Clallam County | Oregon County | 0.50 | Jefferson County | Jefferson County | Bridge right, but the answer passage was never retrieved |
| 120 | Where is the district Ullscarf is part of located in the UK? | county of Cumbria | English Lake District | 0.00 | Lake District | Lake District | Bridge right, but the answer passage was never retrieved |
| 129 | How old do you have to be to buy a handgun in the state where the mouth of the English river is located? | 21 or older. | 21 | 0.50 | Plymouth | Iowa | Wrong bridge: the hop-1 passage was never retrieved |
| 131 | What is the most common source of electricity in the province Provencher is located in? | Manitoba Hydro | hydroelectric power | 0.00 | Manitoba | Manitoba | Bridge right, answer passage shown, but the model read the wrong fact |
| 133 | When did the team Brandon London was a member of last win the Super Bowl? | 2011 | 2008 | 0.00 | Giants | Giants | Bridge right, answer passage retrieved but never shown to the model |
| 134 | Who build the largest ancient Inca empire in the region that American studies is a facet of? | Sapa Inca | Inca | 0.67 | western South America | the Americas | Wrong bridge: the hop-1 passage was never retrieved |
| 137 | When was the record label of Factory Floor established? | 2001 | 2013 | 0.00 | Factory Floor | DFA Records | Wrong bridge: right passage was retrieved but the model picked another entity |
| 140 | When did All Bad singer's Flatline come out? | January 2, 2014 | December 23, 2013 | 0.00 | Bieber | Justin Bieber | Bridge right, answer passage shown, but the model read the wrong fact |
| 141 | When did the country that the entertainer Harrysong was from, become a member of OPEC? | 1971 | the 1970s | 0.00 | Nigeria | Nigeria | Bridge right, but the answer passage was never retrieved |
| 145 | How old do you have to be to buy a handgun in the state where Patty Waters was born? | 21 or older. | 21 | 0.50 | Iowa | Iowa | Partly right: answer wording/span differs from gold |
| 147 | Why was the country where taekwondo is originally from divided at the 38th parallel at the end of WW2? | it divided the country approximately in half | 38th parallel | 0.00 | (none) | Korea | No bridge: hop 1 found nothing usable (hop-1 passage never retrieved) |
| 152 | Who introduced the first microprocessor for the developer of Kentsfield in 1971? | Federico Faggin | Ted Hoff, Federico Faggin, and Stanley Mazor | 0.44 | Intel | Intel | Bridge right, but the answer passage was never retrieved |
| 153 | In what form was the first book from the writer who mentioned The Angel pub published? | Monthly serial | book form | 0.00 | Charles Dickens | Charles Dickens | Bridge right, but the answer passage was never retrieved |
| 154 | What day did the Swapnote publisher unveil the new systems? | October 18, 1985 | December 21, 2011 | 0.00 | December 21, 2011 | Nintendo | Wrong bridge: right passage was retrieved but the model picked another entity |
| 156 | Who is a cast member on the show which Max Hoyland is from? | Tim Phillipps | Calen Mackenzie | 0.00 | Neighbours | Neighbours | Bridge right, answer passage retrieved but never shown to the model |
| 157 | What area does the state where Butterfly Pond locates contain? | Kent County | Providence County | 0.50 | Rhode Island | Rhode Island | Bridge right, but the answer passage was never retrieved |
| 159 | Who was in charge of William Boone's city of birth? | John Cranley | Eugene P. Ruehlmann | 0.00 | Cincinnati | Cincinnati | Bridge right, but the answer passage was never retrieved |
| 160 | When did trolley buses stop running in the birthplace of George Shipley? | 2 October 1966 | 1949 | 0.00 | Newcastle | Newcastle | Bridge right, but the answer passage was never retrieved |
| 164 | During the cold war, what happened to the country where Phnom Aural is found? | Cambodian Civil War | Khmer Rouge victory | 0.00 | Cambodia | Cambodia | Bridge right, answer passage shown, but the model read the wrong fact |
| 165 | Which county borders the county where Rose Township is located? | Indiana County | Jefferson County | 0.50 | Shelby County | Jefferson County | Bridge right, but the answer passage was never retrieved |
| 166 | The old form of the language known by Mahsati is a part of what larger group of languages? | Iranian languages | Indo - Iranian | 0.50 | Old Iranian | Persian | Wrong bridge: the hop-1 passage was never retrieved |
| 168 | What part of Utah is the city Palace of Buddies formed? | Salt Lake City metropolitan area | The City of the Great Salt Lake | 0.60 | Salt Lake City | Salt Lake City | Partly right: answer wording/span differs from gold |
| 172 | What spacecraft used the gravitational pull of the planet that the trophy awarded at the annual Montreal fireworks festival, takes its name from? | Voyager probes | Voyager | 0.67 | Jupiter | Jupiter | Partly right: answer wording/span differs from gold |
| 174 | When does the monsoon season occur in the city which is the fashion capital of India? | the middle of the summer | June till September | 0.00 | New York City | New Delhi | Wrong bridge: the hop-1 passage was never retrieved |
| 175 | What was the name of the hurricane that hit the island and US territory that the creature can be found in? | Hurricane Maria | Darwinius | 0.00 | Darwinius | Puerto Rico | Wrong bridge: the hop-1 passage was never retrieved |
| 176 | When did the city where Palace of Buddies formed become the capitol of Utah? | 1858 | 1856 | 0.00 | Salt Lake City | Salt Lake City | Bridge right, answer passage shown, but the model read the wrong fact |
| 177 | Who built the largest ancient Inca empire in the region that Columbus discovered in 1492? | Sapa Inca | The Inca | 0.67 | the Bahamas | the Americas | Wrong bridge: the hop-1 passage was never retrieved |
| 178 | Where is the city where Helen Hardin was born located? | Bernalillo County, New Mexico | New Mexico | 0.67 | Albuquerque | Albuquerque | Partly right: answer wording/span differs from gold |
| 182 | What new powers did the employer of Melvin Purvis now have? | search a house while the residents are away | Melvin Laird | 0.00 | Melvin Laird | FBI | Wrong bridge: the hop-1 passage was never retrieved |
| 184 | What is the highest court in the jurisdiction of Eurocrats? | The European Court of Justice | (empty) | 0.00 | (none) | European Union | No bridge: hop 1 found nothing usable (hop-1 passage never retrieved) |
| 187 | What is the rate of interreligious marriage in the country where Jonathan Arias was born? | as low as 10% | 10% | 0.40 | Mexico | Mexico | Partly right: answer wording/span differs from gold |
| 190 | When did the first large winter carnival take place in the city where Jean Pelletier died? | 1894 | 1955 | 0.00 | Quebec City | Quebec City | Bridge right, answer passage shown, but the model read the wrong fact |
| 192 | What county is WNSC-FM located in? | York County | Rock Hill | 0.00 | South Carolina | Rock Hill | Wrong bridge: right passage was retrieved but the model picked another entity |
| 193 | Who is the programming language with the DELETE function partially named after? | Michael Widenius's daughter | (empty) | 0.00 | (none) | SQL | No bridge: hop 1 found nothing usable |
| 198 | What company manufactures the 747 model for the company that makes Thomas-Morse MB-3? | Boeing Commercial Airplanes | Boeing | 0.50 | Boeing | Boeing | Partly right: answer wording/span differs from gold |
| 199 | Douré, in Kamsi's country, is in which department? | Zimtenga Department | Doulougou | 0.00 | Doulougou | Burkina Faso | Wrong bridge: right passage was retrieved but the model picked another entity |
| 202 | What was the average family size in the city where Daniel William Coquillett died? | 3.26 | 3.34 | 0.00 | Atlantic City, New Jersey | Atlantic City | Bridge right, answer passage shown, but the model read the wrong fact |
| 203 | When did the headquarters location of Greatview fall? | June 6 | 25 April 1644 | 0.00 | Beijing, China | Beijing | Bridge right, but the answer passage was never retrieved |
| 207 | Where was the opening scene of the TV show that featured Ben Matlock filmed? | Franklin Canyon Park | Summerville | 0.00 | Matlock | Andy Griffith | Wrong bridge: right passage was retrieved but the model picked another entity |
| 208 | Where is Shelby High School located in the state whose motto is a passage from the Bible? | Northern Ohio League | Shelby | 0.00 | Shelby High School | Ohio | Wrong bridge: right passage was retrieved but the model picked another entity |
| 209 | What town is found in the same county as Lorraine? | Belle Fourche | Diamondville | 0.00 | Butte County | Butte County | Bridge right, but the answer passage was never retrieved |
| 211 | What was the final thing shown on the creator of Till Death Do Us Part before it was shut down for the war? | Mickey's Gala Premier | engine | 0.00 | Johnny Speight | BBC | Wrong bridge: right passage was retrieved but the model picked another entity |
| 212 | When did Japan finish invading the country where the last name shin came from? | 1598 | 1945 | 0.00 | Korea | Korea | Bridge right, but the answer passage was never retrieved |
| 213 | The language of work or name of Massis is a type of what? | Unicode block | weekly newspaper | 0.00 | Armenian | Armenian | Bridge right, answer passage shown, but the model read the wrong fact |
| 214 | The seventh mission, of the series of missions that Project Gemini helped recruit experienced people for, launched from Earth on what date? | October 11, 1968 | Blue Gemini | 0.00 | Blue Gemini | Apollo | Wrong bridge: right passage was retrieved but the model picked another entity |
| 218 | When was the spring training home of Rich Thompson's team built? | 1976 | 2014 | 0.00 | Los Angeles Angels | Oakland Athletics | Wrong bridge: right passage was retrieved but the model picked another entity |
| 220 | Who owns the 100+ Station Group that shares its name with the network that plays Blue Collar TV? | Warner Bros. | Time Warner | 0.50 | The WB | The WB | Partly right: answer wording/span differs from gold |
| 223 | When did the U.S. and the country where Winchester is located become allies? | Since 1940 | 1917 | 0.00 | England | England | Bridge right, but the answer passage was never retrieved |
| 224 | Who was the king in 1936 of the country where Freemasonry first spread? | Edward VIII | James VI and I | 0.00 | Scotland | England | Wrong bridge: right passage was retrieved but the model picked another entity |
| 226 | How many times did Danny Philliskirk's team win the Premier League? | 5 | 0 | 0.00 | Blackpool | Chelsea | Wrong bridge: right passage was retrieved but the model picked another entity |
| 229 | What agency consists of the Science Museum in Alberto Lista's birth place? | Spanish National Research Council | Archeological Museum of Seville | 0.00 | Seville | Seville | Bridge right, answer passage retrieved but never shown to the model |
| 231 | Who was married to the artist behind Look to Your Heart? | Barbara Marx | Romy Rosemont | 0.00 | Frank Sinatra | Frank Sinatra | Bridge right, but the answer passage was never retrieved |
| 234 | In what state is the county that the Dodge Center is found in? | Nebraska | Minnesota | 0.00 | Dodge County | Dodge County | Bridge right, answer passage shown, but the model read the wrong fact |
| 236 | When did the team that lost the 2005 Champions League final last win Serie A? | 2010 -- 11 | 1946 - 47 | 0.00 | Liverpool | Milan | Wrong bridge: right passage was retrieved but the model picked another entity |
| 237 | Who is the child of the musician who wrote the lyrics to The Sound of Silence? | Harper Simon | Josette Simon | 0.50 | Paul Simon | Paul Simon | Bridge right, but the answer passage was never retrieved |
| 239 | What is the average family size in the city in which WPGG is licensed? | 3.26 | 3.34 | 0.00 | Atlantic City | Atlantic City | Bridge right, answer passage shown, but the model read the wrong fact |
| 240 | Where is the Rio Grande river located in the country where Norbrook is located? | parish of Portland | Portland | 0.50 | Jamaica | Jamaica | Partly right: answer wording/span differs from gold |
| 242 | Who is a cast member from the show with a character named Jan Brandner? | Valerie Niehaus | Andreas Brucker | 0.00 | Roald Dahl | Verbotene Liebe | Wrong bridge: right passage was retrieved but the model picked another entity |
| 245 | Kanye is credited for the shift away from what specific genre of Height's genre? | gangsta rap | (empty) | 0.00 | (none) | rap | No bridge: hop 1 found nothing usable (hop-1 passage never retrieved) |
| 246 | When did the ocean surrounding Antarctica become a thing? | the 1770s | 30 million years ago | 0.00 | roughly 30 million years ago | Southern Ocean | Wrong bridge: the hop-1 passage was never retrieved |
| 255 | When did Chopin's family relocate to the town of Jan Bialostocki's demise? | October 1810 | Krakowskie Przedmieście | 0.00 | 1827 | Warsaw | Wrong bridge: the hop-1 passage was never retrieved |
| 258 | Where is Sheet Harbour in the state having Great Village? | Halifax Regional Municipality | Nova Scotia | 0.00 | Nova Scotia | Nova Scotia | Bridge right, answer passage shown, but the model read the wrong fact |
| 261 | The modern setting of Blind Detective was ruled by the British Empire starting when? | 1842 | 2013 | 0.00 | British Empire | Hong Kong | Wrong bridge: right passage was retrieved but the model picked another entity |
| 262 | Who is the composer of the piece of which Atys is an instance, named after a beggar? | Johann Christoph Pepusch | Piccinni | 0.00 | Bikini | opera | Wrong bridge: right passage was retrieved but the model picked another entity |
| 264 | Who is the father of the singer-songwriter who had a compilation album called Acoustic? | Alfred Lennon | Julian Lennon | 0.50 | John Lennon | John Lennon | Bridge right, but the answer passage was never retrieved |
| 267 | Who founded the institution that Domenico Crivelli is employed by? | Edward Fisher | King George III | 0.00 | Royal Academy of Music | Royal Academy of Music | Bridge right, answer passage shown, but the model read the wrong fact |
| 272 | When did the most followed person on Instagram join Manchester United? | 2003 | professional football club based in Old Trafford | 0.00 | professional football club based in Old Trafford, Greater Manchester, England | Cristiano Ronaldo | Wrong bridge: the hop-1 passage was never retrieved |
| 273 | when did England colonize the country where Freetown is located? | 1666 | 1808 | 0.00 | Sierra Leone | Bahamas | Wrong bridge: right passage was retrieved but the model picked another entity |
| 274 | What month did Einstein's performer win? | September | May | 0.00 | Kelly Clarkson | Kelly Clarkson | Bridge right, but the answer passage was never retrieved |
| 277 | When did the performer of Stronger release her first album? | 1999 | 2003 | 0.00 | Kelly Clarkson | Britney Spears | Wrong bridge: right passage was retrieved but the model picked another entity |
| 280 | Who was targeted for defrauding the electoral process, in the state where Pete King was born? | the Republicans are willing to do things that are unethical to manipulate elections | Senator Boxer | 0.00 | Ohio | Ohio | Bridge right, but the answer passage was never retrieved |
| 281 | When did Baird cease showing its programs on the network which broadcasted Elizabeth R? | June 1932 | 23 May 1995 | 0.00 | BBC2 | BBC | Wrong bridge: right passage was retrieved but the model picked another entity |
| 282 | What college did Seemabaddha's composer attend? | Visva-Bharati University | Mani Shankar Mukherjee | 0.00 | Mani Shankar Mukherjee | Satyajit Ray | Wrong bridge: right passage was retrieved but the model picked another entity |
| 283 | What language is used by the director of Tiffany Memorandum? | Italian | English | 0.00 | Sergio Grieco | Sergio Grieco | Bridge right, answer passage shown, but the model read the wrong fact |
| 284 | What did the war that where the Desert Patrol Vehicle first fought inadvertently do in the early 1990s? | radicalize the Islamist movement | radicalized the Islamist movement | 0.67 | Gulf War | Gulf War | Partly right: answer wording/span differs from gold |
| 290 | How did David Rossi's employer respond to requests to release information beyond the initial press release? | The FBI refused | refused | 0.67 | FBI | FBI | Partly right: answer wording/span differs from gold |
| 291 | The 7, made by the company that also built the AMR 35, is part of which series? | supermini | Renault 7 | 0.00 | Renault | Renault | Bridge right, answer passage shown, but the model read the wrong fact |
| 292 | Which city in the province where Tokyo Rose was founded borders Riviere-Verte? | Edmundston | New Brunswick | 0.00 | New Jersey | New Brunswick | Bridge right, answer passage shown, but the model read the wrong fact |
| 294 | Who is the programming language that has the WHERE clause partially named after? | Michael Widenius's daughter | Codd | 0.00 | SQL | SQL | Bridge right, but the answer passage was never retrieved |
| 300 | On what network can you find the Tonight show that bears the name of the channel that broadcast Market Wrap? | CNBC Asia | CNBC | 0.67 | CNBC | CNBC | Partly right: answer wording/span differs from gold |
