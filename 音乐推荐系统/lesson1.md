## 前言
任何的机器学习算法解决问题，首先要考虑的是数据，数据从何而来？
* 对于酷狗音乐/网易音乐这样的企业而言，用户的收藏和播放数据是可以直接获得的
* 我们找一个取巧的方式，包含用户音乐兴趣信息，同时又可以获取的数据是什么？ 对的，是歌单信息

![](./pic/163_music.png)

## 一、原始数据

利用爬虫技术获取网易云音乐歌单，存储为json格式，压缩后的全量数据共计16G，解压后超过50G。由于数据量太大，且代码中用不到这部分原始数据，故没有提供原始数据，此处仅展示原始数据的格式：

每个歌单的格式

```
{
    "result": {
        "id": 111450065,
        "status": 0,
        "commentThreadId": "A_PL_0_111450065",
        "trackCount": 120,
        "updateTime": 1460164523907,
        "commentCount": 227,
        "ordered": true,
        "anonimous": false,
        "highQuality": false,
        "subscribers": [],
        "playCount": 687070,
        "trackNumberUpdateTime": 1460164523907,
        "createTime": 1443528317662,
        "name": "带本书去旅行吧,人生最美好的时光在路上。",
        "cloudTrackCount": 0,
        "shareCount": 149,
        "adType": 0,
        "trackUpdateTime": 1494134249465,
        "userId": 39256799,
        "coverImgId": 3359008023885470,
        "coverImgUrl": "http://p1.music.126.net/2ZFcuSJ6STR8WgzkIi2U-Q==/3359008023885470.jpg",
        "artists": null,
        "newImported": false,
        "subscribed": false,
        "privacy": 0,
        "specialType": 0,
        "description": "现在是一年中最美好的时节，世界上很多地方都不冷不热，有湛蓝的天空和清冽的空气，正是出游的好时光。长假将至，你是不是已经收拾行装准备出发了？行前焦虑症中把衣服、洗漱用品、充电器之类东西忙忙碌碌地丢进箱子，打进背包的时候，我打赌你肯定会留个位置给一位好朋友：书。不是吗？不管是打发时间，小读怡情，还是为了做好攻略备不时之需，亦或是为了小小地装上一把，你都得有一本书傍身呀。读大仲马，我是复仇的伯爵；读柯南道尔，我穿梭在雾都的暗夜；读村上春树，我是寻羊的冒险者；读马尔克斯，目睹百年家族兴衰；读三毛，让灵魂在撒哈拉流浪；读老舍，嗅着老北京的气息；读海茵莱茵，于科幻狂流遨游；读卡夫卡，在城堡中审判……读书的孩子不会孤单，读书的孩子永远幸福。",
        "subscribedCount": 10882,
        "totalDuration": 0,
        "tags": [
            "旅行",
            "钢琴",
            "安静"]
        "creator": {
            "followed": false,
            "remarkName": null,
            "expertTags": [
                "古典",
                "民谣",
                "华语"
            ],
            "userId": 39256799,
            "authority": 0,
            "userType": 0,
            "gender": 1,
            "backgroundImgId": 3427177752524551,
            "city": 360600,
            "mutual": false,
            "avatarUrl": "http://p1.music.126.net/TLRTrJpOM5lr68qJv1IyGQ==/1400777825738419.jpg",
            "avatarImgIdStr": "1400777825738419",
            "detailDescription": "",
            "province": 360000,
            "description": "",
            "birthday": 637516800000,
            "nickname": "有梦人生不觉寒",
            "vipType": 0,
            "avatarImgId": 1400777825738419,
            "defaultAvatar": false,
            "djStatus": 0,
            "accountStatus": 0,
            "backgroundImgIdStr": "3427177752524551",
            "backgroundUrl": "http://p1.music.126.net/LS96S_6VP9Hm7-T447-X0g==/3427177752524551.jpg",
            "signature": "漫无目的的乱听，听着，听着，竟然灵魂出窍了。更多精品音乐美图分享请加我微信hu272367751。微信是我的精神家园，有我最真诚的分享。",
            "authStatus": 0}
        "tracks": [{歌曲1},{歌曲2}, ...]
     }
}
```

每首歌曲的格式

```
{
	"id": 29738501,
	"name": "跟着你到天边 钢琴版",
	"duration": 174001,
	"hearTime": 0,
	"commentThreadId": "R_SO_4_29738501",
	"score": 40,
	"mvid": 0,
	"hMusic": null,
	"disc": "",
	"fee": 0,
	"no": 1,
	"rtUrl": null,
	"ringtone": null,
	"rtUrls": [],
	"rurl": null,
	"status": 0,
	"ftype": 0,
	"mp3Url": "http://m2.music.126.net/vrVa20wHs8iIe0G8Oe7I9Q==/3222668581877701.mp3",
	"audition": null,
	"playedNum": 0,
	"copyrightId": 0,
	"rtype": 0,
	"crbt": null,
	"popularity": 40,
	"dayPlays": 0,
	"alias": [],
	"copyFrom": "",
	"position": 1,
	"starred": false,,
	"starredNum": 0
	"bMusic": {
	    "name": "跟着你到天边 钢琴版",
	    "extension": "mp3",
	    "volumeDelta": 0.0553125,
	    "sr": 44100,
	    "dfsId": 3222668581877701,
	    "playTime": 174001,
	    "bitrate": 96000,
	    "id": 52423394,
	    "size": 2089713
	},
	"lMusic": {
	    "name": "跟着你到天边 钢琴版",
	    "extension": "mp3",
	    "volumeDelta": 0.0553125,
	    "sr": 44100,
	    "dfsId": 3222668581877701,
	    "playTime": 174001,
	    "bitrate": 96000,
	    "id": 52423394,
	    "size": 2089713
	},
	"mMusic": {
	    "name": "跟着你到天边 钢琴版",
	    "extension": "mp3",
	    "volumeDelta": -0.000265076,
	    "sr": 44100,
	    "dfsId": 3222668581877702,
	    "playTime": 174001,
	    "bitrate": 128000,
	    "id": 52423395,
	    "size": 2785510
	},
	"artists": [
	    {
		"img1v1Url": "http://p1.music.126.net/6y-UleORITEDbvrOLV0Q8A==/5639395138885805.jpg",
		"name": "群星",
		"briefDesc": "",
		"albumSize": 0,
		"img1v1Id": 0,
		"musicSize": 0,
		"alias": [],
		"picId": 0,
		"picUrl": "http://p1.music.126.net/6y-UleORITEDbvrOLV0Q8A==/5639395138885805.jpg",
		"trans": "",
		"id": 122455
	    }
	],
	"album": {
	    "id": 3054006,
	    "status": 2,
	    "type": null,
	    "tags": "",
	    "size": 69,
	    "blurPicUrl": "http://p1.music.126.net/2XLMVZhzVZCOunaRCOQ7Bg==/3274345629219531.jpg",
	    "copyrightId": 0,
	    "name": "热门华语248",
	    "companyId": 0,
	    "songs": [],
	    "description": "",
	    "pic": 3274345629219531,
	    "commentThreadId": "R_AL_3_3054006",
	    "publishTime": 1388505600004,
	    "briefDesc": "",
	    "company": "",
	    "picId": 3274345629219531,
	    "alias": [],
	    "picUrl": "http://p1.music.126.net/2XLMVZhzVZCOunaRCOQ7Bg==/3274345629219531.jpg",
	    "artists": [
		{
		    "img1v1Url": "http://p1.music.126.net/6y-UleORITEDbvrOLV0Q8A==/5639395138885805.jpg",
		    "name": "群星",
		    "briefDesc": "",
		    "albumSize": 0,
		    "img1v1Id": 0,
		    "musicSize": 0,
		    "alias": [],
		    "picId": 0,
		    "picUrl": "http://p1.music.126.net/6y-UleORITEDbvrOLV0Q8A==/5639395138885805.jpg",
		    "trans": "",
		    "id": 122455
		}
	    ],
	    "artist": {
		"img1v1Url": "http://p1.music.126.net/6y-UleORITEDbvrOLV0Q8A==/5639395138885805.jpg",
		"name": "",
		"briefDesc": "",
		"albumSize": 0,
		"img1v1Id": 0,
		"musicSize": 0,
		"alias": [],
		"picId": 0,
		"picUrl": "http://p1.music.126.net/6y-UleORITEDbvrOLV0Q8A==/5639395138885805.jpg",
		"trans": "",
		"id": 0
	    }
	}
}
```

## 二、解析基础歌单数据

抽取 **歌单名称，歌单id，收藏数，所属分类** 4个歌单维度的信息 

抽取 **歌曲id，歌曲名，歌手，歌曲热度** 等4个维度信息歌曲的信息

组织成如下格式：
```
漫步西欧小镇上##小语种,旅行##69413685##474	18682332::Wäg vo dir::Joy Amelie::70.0	4335372::Only When I Sleep::The Corrs::60.0	2925502::Si Seulement::Lynnsha::100.0	21014930::Tu N'As Pas Cherché...::La Grande Sophie::100.0	20932638::Du behöver aldrig mer vara rädd::Lasse Lindh::25.0	17100518::Silent Machine::Cat Power::60.0	3308096::Kor pai kon diew : ชอไปคนเดียว::Palmy::5.0	1648250::les choristes::Petits Chanteurs De Saint Marc::100.0	4376212::Paddy's Green Shamrock Shore::The High Kings::25.0	2925400::A Todo Color::Las Escarlatinas::95.0	19711402::Comme Toi::Vox Angeli::75.0	3977526::Stay::Blue Cafe::100.0	2538518::Shake::Elize::85.0	2866799::Mon Ange::Jena Lee::85.0	5191949::Je M'appelle Helene::Hélène Rolles::85.0	20036323::Ich Lieb' Dich Immer Noch So Sehr::Kate & Ben::100.0
```

原始数据解析后的基础歌单数据 163_music_playlist.txt 最终要转换为surprise库支持的格式，故实际上用不到，此处没有提供；其格式与解析后的华语流行音乐歌单数据一样，如下所示：


```python
!head ./data/popular.playlist
```

## 三、将歌单数据解析为surprise库支持的格式


```python
# 解析成userid itemid rating timestamp行格式

import json
import sys

def is_null(s): 
    return len(s.split(","))>2

def parse_song_info(song_info):
    try:
        song_id, name, artist, popularity = song_info.split(":::")
        #return ",".join([song_id, name, artist, popularity])
        return ",".join([song_id,"1.0",'1300000'])
    except Exception as e:
        #print e
        #print song_info
        return ""

def parse_playlist_line(in_line):
    try:
        contents = in_line.strip().split("\t")
        name, tags, playlist_id, subscribed_count = contents[0].split("##")
        songs_info = map(lambda x:playlist_id+","+parse_song_info(x), contents[1:])
        songs_info = filter(is_null, songs_info)
        return "\n".join(songs_info)
    except Exception as e:
        print(e)
        return False
        

def parse_file(in_file, out_file):
    out = open(out_file, 'w')
    for line in open(in_file):
        result = parse_playlist_line(line)
        if(result):
            out.write(result.strip()+"\n")
    out.close()
```

将华语流行音乐歌单解析为suprise格式


```python
path = "./data/output/popular/"
parse_file("./data/popular.playlist", path+"popular_music_suprise_format.txt")
```

## 四、保存 歌单id=>歌单名 和 歌曲id=>歌曲名 的字典


```python
import pickle
import sys

def parse_playlist_get_info(in_line, playlist_dic, song_dic):
    contents = in_line.strip().split("\t")
    name, tags, playlist_id, subscribed_count = contents[0].split("##")
    playlist_dic[playlist_id] = name
    for song in contents[1:]:
        try:
            song_id, song_name, artist, popularity = song.split(":::")
            song_dic[song_id] = song_name+"\t"+artist
        except:
            print("song format error")
            print(song+"\n")

def parse_file(in_file, out_playlist, out_song):
    #从歌单id到歌单名称的映射字典
    playlist_dic = {}
    #从歌曲id到歌曲名称的映射字典
    song_dic = {}
    for line in open(in_file):
        parse_playlist_get_info(line, playlist_dic, song_dic)
    #把映射字典保存在二进制文件中
    pickle.dump(playlist_dic, open(out_playlist,"wb")) 
    #可以通过 playlist_dic = pickle.load(open("playlist.pkl","rb"))重新载入
    pickle.dump(song_dic, open(out_song,"wb"))
```


```python
parse_file("./data/popular.playlist", path+"popular_playlist.pkl", path+"popular_song.pkl")
```


```python

```


```python

```
