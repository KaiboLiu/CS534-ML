	
###11/21/2016

More languages added

|#|name|language|description|  
|:---:|:----:|:----:|:---  
|1|[cmn](http://tatoeba.org/eng/sentences/with_audio/cmn "Chinese")| Chinese(Mandarin)| female:`500`|  
|2|[eng](http://tatoeba.org/eng/sentences/with_audio/eng "English")| English| male:`500`|  
|2|[engStory]('wutheringheights' "English")|English| female, splitted into 688 wavs, from Eugene's story audio|  
|3|[deu](http://tatoeba.org/eng/sentences/with_audio/deu "German,")| German| male:400|  
|4|[fra](http://tatoeba.org/eng/sentences/with_audio/fra "French,")| French| male:400|  
|5|[jpn](http://tatoeba.org/eng/sentences/with_audio/jpn "Japanes")| Japanese| female:400|  
|6|[rus](http://tatoeba.org/eng/sentences/with_audio/rus "Russian")| Russian| male:400|  

We can custom the distribution and percentage of data samples from each language appeared in our training and dev dataset, like ingredient from a chef.  
See `line 64-65` in `diyDataset.py`, and the comments above them. I offer 6 languages and 7 type of `.wav` files.

Any of you can modify `n_list_train` and `n_list_dev` with the idea how many elements from the corresponding language should be added in new training and dev data. What you need to do is typing the number, it can randomly choose the amount you want from one exact language into dataset and copy `.wav` into a new folder `./diyDataset`.

---
###11/15/2016

Dataset for final project was shared in Google Drive via email.   
If any question, please feel free to contact Kaibo.  

The version of dataset on 11/15/2016 is extracted from tatoeba.org.  
I use `AudioCrawl_batch.py` to crawl mp3 files and store their names. Then I use a free software to convert them into wav files.  
This dataset contains two languages: eng and chn/cmn, consisted of 600 training files+200 testing files, with 48000 sampling rate and mono channel.  

I got another dataset from the source of voxforge, they are direct wav files but only English. I can offer them or expand the tatoeba one if you need more data.



Kaibo