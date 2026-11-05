# Robot Learning ve VLA Alanına Giriş

Bu notun amacı, son günlerde karşılaşılan terimleri tek tek ezberletmek değil, alanın zihinsel haritasını kurmaktır.

Önce kavramları doğru katmanlara ayıracağız. Sonra OpenVLA, OpenVLA-OFT, LIBERO, MuJoCo, ROS, LeRobot, Microduck gibi isimleri bu yapının içine yerleştireceğiz.

---

# 1. En büyük resim

Robot learning problemini en basit haliyle şöyle düşün:

```text
OBSERVATION
    |
    v
  POLICY
    |
    v
 ACTION
    |
    v
WORLD CHANGES
    |
    v
NEW OBSERVATION
```

Bir robot dünyayı gözlemler, bir karar verir, bir action üretir, bu action dünyayı değiştirir ve robot yeni bir observation alır.

Bu döngü sürekli devam eder.

Bir görüntü sınıflandırma problemi ile karşılaştırırsak:

```text
Image
  |
  v
 CNN
  |
  v
Class
```

Robot learning'de ise:

```text
Image + Robot State + Language
             |
             v
           Policy
             |
             v
           Action
```

Asıl temel kavram burada **policy** kelimesidir.

Policy, robotun mevcut gözlemlerine bakarak ne yapacağına karar veren fonksiyondur.

---

# 2. Robot Learning nedir?

Robot learning, robot davranışını elle yazılmış kurallarla tanımlamak yerine veriden öğrenmeye çalışan alandır.

Örneğin klasik bir sistem şöyle olabilir:

```text
if object_is_red:
    move_arm_left()
if gripper_distance < threshold:
    close_gripper()
```

Robot learning yaklaşımında ise sistem çok sayıda örnekten şu ilişkiyi öğrenmeye çalışır:

```text
observation -> action
```

Observation içinde şunlar olabilir:

- RGB kamera görüntüsü
- depth görüntüsü
- robot eklem açıları
- gripper durumu
- force / torque bilgisi
- dil komutu
- önceki action'lar

Action ise örneğin:

- kolu 2 cm sola götür
- bileği döndür
- gripper'ı kapat
- robot tabanını ileri sür

olabilir.

---

# 3. Embodied AI nedir?

Embodied AI daha geniş bir kavramdır.

Bir AI sisteminin fiziksel veya simüle edilmiş bir dünyada algılayıp hareket ederek görev çözmesini inceler.

Basit bir hiyerarşi:

```text
Artificial Intelligence
|
+-- Embodied AI
    |
    +-- Robot Learning
        |
        +-- Robot Policies
            |
            +-- Vision-Language-Action models
```

VLA bu alanın tamamı değildir.

VLA, robot policy üretmek için kullanılan modern yaklaşımlardan biridir.

---

# 4. Bir robot learning sisteminin temel parçaları

Alanı anlamanın en kolay yolu şu bileşenleri birbirinden ayırmaktır.

## 4.1 Dünya

Robotun içinde bulunduğu ortam.

Bu ortam:

- gerçek dünya
- simülasyon

olabilir.

Örnek:

```text
masa
kupa
çekmece
robot kolu
kamera
```

---

## 4.2 Task

Robot ne yapmak zorunda?

Örneğin:

```text
"Pick up the red cup."
"Open the drawer."
"Put the apple into the bowl."
```

Task bazen açık bir dil komutu olabilir.

Bazen de training sırasında dolaylı olarak tanımlanmış olabilir.

---

## 4.3 Observation

Robot dünyanın hangi kısmını görüyor?

Örnek:

```text
RGB image
depth image
joint positions
gripper state
language instruction
previous actions
```

Robot her zaman tüm dünyanın durumunu bilmez.

Sadece sensörlerinden gelen observation'a sahiptir.

---

## 4.4 Policy

Policy sistemin karar veren kısmıdır.

```text
observation
    |
    v
 policy
    |
    v
 action
```

Machine learning tarafındaki esas model buradadır.

Bir VLA da bir tür policy'dir.

---

## 4.5 Action

Robotun uygulayacağı fiziksel komut.

Örnek:

```text
delta_x = +0.02
delta_y = -0.01
delta_z = 0.00
gripper = close
```

Bazı sistemler doğrudan düşük seviyeli motor command üretir.

Bazıları daha yüksek seviyeli action üretir.

---

# 5. Robot nasıl öğrenir?

Robot öğrenmesi için data gerekir.

Data farklı yollarla üretilebilir.

## 5.1 Human Demonstration

İnsan robota görevi gösterir.

Örneğin teleoperation ile:

```text
camera image
robot state
human action
```

kaydedilir.

Ortaya trajectory çıkar.

---

## 5.2 Teleoperation

İnsan joystick, VR controller veya başka bir arayüzle robotu kontrol eder.

Her zaman adımında şu çift kaydedilir:

```text
observation_t -> action_t
```

Bunlardan training dataset oluşur.

---

## 5.3 Simulation Data

Robot görevleri simülasyonda çalıştırılabilir.

Avantaj:

- gerçek robot kırılmaz
- veri daha hızlı üretilebilir
- environment kolay değiştirilebilir
- aynı görev binlerce kez tekrar edilebilir

Dezavantaj:

- simülasyon ile gerçek dünya birebir aynı değildir

Bu fark:

**sim-to-real gap**

olarak adlandırılır.

---

# 6. Imitation Learning

En temel robot learning paradigmalarından biridir.

Amaç:

> İnsan veya uzman ne yaptıysa onu öğren.

Dataset:

```text
observation_1 -> action_1
observation_2 -> action_2
observation_3 -> action_3
...
```

Model bunlardan:

```text
observation -> action
```

eşleşmesini öğrenir.

---

# 7. Behavior Cloning

Imitation Learning'in en temel biçimlerinden biridir.

Supervised learning gibi düşünülebilir.

```text
X = observation
y = expert action
```

Model:

```text
observation -> predicted action
```

öğrenir.

Bu, robot learning'e giriş için en önemli kavramlardan biridir.

---

# 8. Reinforcement Learning

Reinforcement Learning'de robot doğrudan doğru action label'larıyla öğrenmek zorunda değildir.

Robot:

```text
state
  |
action
  |
reward
```

döngüsü üzerinden öğrenir.

Örnek:

```text
cup successfully grasped -> +1
cup dropped -> -1
```

Ama gerçek robotlarda RL pahalı olabilir çünkü çok fazla deneme gerekir.

Bu nedenle robot manipulation alanında imitation learning çok önemlidir.

---

# 9. Manipulation nedir?

VLA çalışmalarının büyük kısmı robot manipulation üzerine kuruludur.

Manipulation:

Robotun fiziksel nesnelerle etkileşmesi.

Örnek görevler:

```text
pick
place
push
open
close
insert
pour
fold
stack
```

Bir VLA paper gördüğünde büyük ihtimalle manipulation görevleri ile karşılaşacaksın.

---

# 10. VLA nedir?

VLA:

**Vision-Language-Action**

Temel fikir:

```text
Vision
  +
Language
  +
Robot State
  |
  v
 VLA
  |
  v
Action
```

Örneğin:

```text
camera image:
  red mug + blue mug

instruction:
  "pick up the blue mug"

robot state:
  arm position

        |
        v

       VLA

        |
        v

robot action
```

Buradaki kritik fikir şudur:

VLA yalnızca görüntüyü anlamamalı.

Dil komutunu da robot davranışına bağlamalıdır.

Buna:

**language grounding**

denir.

---

# 11. VLM ile VLA farkı

VLM:

Vision-Language Model

Örneğin:

```text
image + question -> text answer
```

VLA:

Vision-Language-Action

```text
image + instruction -> robot action
```

Yani VLA'nın çıkışı yalnızca dil değildir.

Robot davranışıdır.

---

# 12. VLA modellerine örnekler

VLA bir model adı değil, bir model ailesidir.

Örnekler:

- RT-2
- OpenVLA
- Octo
- π0
- π0-Fast
- NORA
- UniVLA
- WorldVLA

Hepsi aynı mimariye sahip değildir.

Ama ortak amaçları, vision ve language bilgisini robot action üretiminde kullanmaktır.

---

# 13. OpenVLA nedir?

OpenVLA açık kaynaklı bir VLA modelidir.

En basit zihinsel konum:

```text
Robot Learning
    |
    +-- VLA
        |
        +-- OpenVLA
```

Yani:

```text
VLA = kategori
OpenVLA = bu kategorideki belirli bir model
```

OpenVLA farklı robot manipulation görevlerinde fine-tune edilip değerlendirilebilir.

---

# 14. OpenVLA-OFT nedir?

OpenVLA-OFT ayrı bir VLA paradigması değildir.

OpenVLA'nın daha etkili fine-tuning ve training yöntemleriyle geliştirilmiş bir versiyonudur.

Kabaca:

```text
OpenVLA
   |
   | improved fine-tuning recipe
   v
OpenVLA-OFT
```

Buradaki OFT yaklaşımı OpenVLA'nın özellikle downstream robot manipulation görevlerinde daha iyi performans vermesini hedefler.

Dolayısıyla OpenVLA ile LIBERO üzerine çalışırken OpenVLA-OFT paper'ının karşına çıkması normaldir.

Çünkü soru genellikle şudur:

```text
OpenVLA LIBERO'da nasıl daha iyi fine-tune edilir?
```

OpenVLA-OFT bu soruya verilen önemli cevaplardan biridir.

---

# 15. Benchmark nedir?

Benchmark model değildir.

Benchmark bir sınav sistemidir.

Bir benchmark genellikle şunları tanımlar:

- environment
- görevler
- başlangıç durumları
- demonstrasyonlar
- evaluation protocol
- success metric

Bir model benchmark üzerinde değerlendirilir.

---

# 16. LIBERO nedir?

LIBERO bir robot learning benchmark ailesidir.

Özellikle manipulation görevleri içerir.

Basit zihinsel model:

```text
OpenVLA = öğrenci
LIBERO  = sınav
```

LIBERO içinde farklı task suite'leri bulunur.

Bunlar robotun farklı generalization özelliklerini ölçmeyi amaçlar.

Örneğin:

- object
- spatial
- goal
- long-horizon

gibi farklı görev kümeleri vardır.

---

# 17. LIBERO-Plus nedir?

LIBERO-Plus yeni bir VLA modeli değildir.

LIBERO benchmark'ının robustness açısından daha zorlaştırılmış bir değerlendirme yaklaşımıdır.

Amaç:

> Model gerçekten görevi mi öğrendi, yoksa benchmark'taki kolay shortcut'ları mı kullanıyor?

Örneğin environment şu şekillerde değiştirilebilir:

- language perturbation
- object appearance perturbation
- background değişimi
- camera değişimi
- distractor object
- farklı spatial configuration

Bu sayede modelin gerçekten genelleşip genelleşmediği ölçülür.

---

# 18. OpenVLA-OFT ve language grounding problemi

LIBERO-Plus paper'ındaki ilginç bulgulardan biri şudur:

Bazı görevlerde OpenVLA-OFT'nin language instruction geçersiz hale getirilse bile performansı büyük ölçüde korunabiliyor.

Bu ilk bakışta çok garip görünür.

Çünkü VLA'nın amacı:

```text
Vision + Language -> Action
```

iken model sanki şu şekilde davranıyor olabilir:

```text
Vision -> Action
```

ve language kısmını çok az kullanıyor olabilir.

Bu durum özellikle benchmark'ta her environment belirli bir task ile çok güçlü şekilde eşleşiyorsa ortaya çıkabilir.

Örneğin model environment görüntüsünden zaten hangi görevin yapılacağını tahmin edebiliyorsa language instruction gereksiz hale gelebilir.

Bu bir tür:

**shortcut learning**

problemi olabilir.

---

# 19. Base OpenVLA ile OpenVLA-OFT'yi karıştırmamak gerekir

OpenVLA-OFT ile ilgili bulunan language insensitivity sonucu doğrudan base OpenVLA'ya genellenemez.

Çünkü fine-tuning yöntemi modelin behavior distribution'ını ciddi şekilde değiştirebilir.

Önemli ayrım:

```text
OpenVLA
   !=
OpenVLA-OFT
```

Aynı temel model ailesinden gelseler bile downstream fine-tuning sonrası farklı davranabilirler.

Bu yüzden paper okurken modelin tam checkpoint ve training recipe'sine dikkat etmek gerekir.

---

# 20. Generalization neden bu kadar önemli?

Bir robot training sırasında yalnızca gördüğü durumları ezberlerse gerçek dünyada işe yaramaz.

Örneğin:

Training:

```text
red cup
wooden table
camera angle A
```

Test:

```text
green cup
white table
camera angle B
```

Model hâlâ görevi yapabiliyor mu?

Bu generalization problemidir.

---

# 21. Robot learning'deki temel araştırma problemleri

Model isimlerinden önce problemleri bilmek daha faydalıdır.

## 21.1 Manipulation

Robot nesnelerle fiziksel olarak nasıl etkileşir?

---

## 21.2 Generalization

Training sırasında görülmeyen durumlarda çalışabiliyor mu?

---

## 21.3 Language Grounding

Robot language instruction'ı gerçekten kullanıyor mu?

Örnek:

```text
red cup + blue cup

"pick up the red cup"
```

Robot gerçekten "red" kelimesine göre mi seçim yapıyor?

---

## 21.4 Robustness

Environment biraz değişince model bozuluyor mu?

Örnek:

- lighting
- background
- camera position
- object texture
- distractor object
- instruction paraphrase

---

## 21.5 Long-Horizon Tasks

Tek action değil, uzun bir action zinciri gerekir.

Örnek:

```text
open fridge
   |
find apple
   |
pick apple
   |
close fridge
   |
put apple on table
```

Hatalar zaman içinde birikir.

---

## 21.6 Reactive Behavior

Robot environment'ta beklenmedik bir olay olduğunda davranışını değiştirebilir mi?

Örneğin:

```text
instruction:
"Take the apples to the kitchen."

robot carries bag

bag tears

apple falls
```

Robot bunu fark edip planını değiştirebiliyor mu?

Bu güncel VLA sistemleri için zor bir problemdir.

---

## 21.7 Temporal Reasoning

Robot yalnızca mevcut frame'e değil geçmişe de ihtiyaç duyabilir.

Örneğin:

```text
robot drawer'ı biraz önce açtı
camera şu an drawer'ı tam görmüyor
```

Robot geçmiş action'larını ve observation'larını hatırlamalıdır.

---

## 21.8 Planning

Bazı görevlerde doğrudan action üretmek yeterli değildir.

Robot önce bir plan oluşturmalıdır.

Örneğin:

```text
Goal:
make coffee

Plan:
1. find cup
2. pick cup
3. place under machine
4. press button
5. wait
6. retrieve cup
```

---

# 22. Simulation nedir?

Gerçek robot kullanmadan robot environment'ını bilgisayarda çalıştırmaktır.

Simülasyonda:

- robot
- fizik
- kamera
- objeler
- collision
- gravity
- friction

modellenebilir.

Bu alan için önemli simülatörlerden bazıları:

- MuJoCo
- Isaac Sim
- PyBullet
- Gazebo

---

# 23. MuJoCo nedir?

MuJoCo bir physics engine'dir.

Açılımı:

**Multi-Joint dynamics with Contact**

Robot simulation için çok yaygın kullanılır.

MuJoCo şunları simüle eder:

- rigid body dynamics
- joint movement
- contact
- friction
- collision
- gravity

Önemli:

```text
MuJoCo != VLA
MuJoCo != benchmark
MuJoCo != robot learning algorithm
```

MuJoCo sadece fiziksel dünyayı simüle eden altyapıdır.

---

# 24. Benchmark ile Simulator aynı şey değildir

Bu ayrım çok önemlidir.

Simulator:

```text
physics + world
```

Benchmark:

```text
tasks + evaluation protocol
```

Örneğin:

```text
MuJoCo -> physics engine
LIBERO -> benchmark
```

Bir benchmark MuJoCo gibi bir simulator üzerinde çalışabilir.

Ama aynı şey değildir.

---

# 25. ROS nedir?

ROS:

**Robot Operating System**

İsmine rağmen klasik anlamda bir operating system değildir.

Robotik yazılım bileşenlerinin birbirleriyle haberleşmesini sağlayan middleware / framework yapısıdır.

Örneğin:

```text
camera node
     |
     v
vision model
     |
     v
VLA policy
     |
     v
robot controller
     |
     v
motors
```

Bu bileşenler ROS üzerinden mesajlaşabilir.

ROS özellikle gerçek robot sistemlerinde çok önemlidir.

---

# 26. ROS2

Bugün yeni robotik projelerde ROS2 daha yaygın tercih edilmektedir.

ROS2:

- distributed communication
- real-time support
- multi-machine systems
- modern middleware

konularında ROS1'e göre daha güçlüdür.

Yeni başlayan biri için genellikle ROS2 öğrenmek daha mantıklıdır.

---

# 27. LeRobot nedir?

LeRobot, Hugging Face tarafından geliştirilen açık kaynak robot learning ekosistemidir.

Amacı robot learning workflow'unu kolaylaştırmaktır.

Şunları içerir:

- robot datasets
- data collection
- teleoperation
- policy training
- evaluation
- hardware integrations

LeRobot'u şu şekilde düşünmek daha doğru:

```text
robot learning toolbox / ecosystem
```

VLA modeli değildir.

Simulator değildir.

Benchmark da değildir.

---

# 28. Robomimic nedir?

Robomimic robot imitation learning için kullanılan açık kaynak framework'lerden biridir.

Özellikle demonstration dataset'leri üzerinden policy training yapmak için kullanılır.

LeRobot ile aynı şey değildir ama benzer problem alanında araç sağlar.

---

# 29. Microduck nedir?

Microduck gerçek fiziksel robot platformlarından biridir.

Bu tür robotlar küçük ölçekli deneyler ve eğitim için kullanılabilir.

Zihinsel olarak:

```text
Microduck = hardware
```

Onun üzerinde:

- kamera
- motor
- robot arm
- gripper
- onboard computer

gibi bileşenler bulunabilir.

Bir VLA modelini gerçek robotta test etmek istediğinde bu tür hardware platformlarına ihtiyaç duyarsın.

---

# 30. Robot Hardware örnekleri

Robot manipulation araştırmalarında sık karşılaşılan robotlar:

- Franka Panda
- UR5 / UR10
- WidowX
- ALOHA
- xArm
- Kinova
- Fetch

Bunlar VLA modeli değildir.

Fiziksel robot platformlarıdır.

---

# 31. ALOHA nedir?

ALOHA özellikle bimanual manipulation araştırmalarında kullanılan robot sistemi olarak önemlidir.

İki robot kolu ile insan demonstrasyonlarının toplanması ve imitation learning için kullanılabilir.

Mobile ALOHA ise mobil platform + dual arm yaklaşımını birleştirir.

---

# 32. Dataset nedir?

Robot learning dataset'i genellikle yalnızca image-label yapısında değildir.

Bir trajectory şöyle olabilir:

```text
episode_001

t=0
image
robot_state
action

t=1
image
robot_state
action

t=2
image
robot_state
action
```

Bir episode genellikle bir görevin baştan sona çalıştırılmasını temsil eder.

---

# 33. Trajectory nedir?

Trajectory:

Robotun zaman boyunca yaptığı hareketlerin kaydıdır.

Kabaca:

```text
(o1, a1)
(o2, a2)
(o3, a3)
...
(oT, aT)
```

Burada:

```text
o = observation
a = action
```

Robot learning paper'larında çok sık karşılaşılır.

---

# 34. Episode nedir?

Episode genellikle tek bir task attempt anlamına gelir.

Örneğin:

```text
Task:
pick up red cup

Episode starts

robot moves
robot grasps cup
robot lifts cup

Episode ends
```

Başarılı veya başarısız olabilir.

---

# 35. Success Rate

Robot benchmark'larında en temel metric'lerden biridir.

Örneğin:

```text
100 task attempt
87 successful

success rate = 87%
```

Ama yüksek success rate her zaman gerçek intelligence anlamına gelmez.

LIBERO-Plus gibi çalışmalar tam olarak bunu sorgular.

---

# 36. Shortcut Learning

Model task'ın gerçek nedenini öğrenmek yerine dataset'teki kolay korelasyonları kullanabilir.

Örneğin:

```text
environment A -> always pick red cup
environment B -> always open drawer
```

Model language instruction'ı kullanmadan environment görüntüsünden task'ı tahmin edebilir.

Sonuç:

```text
benchmark score = high
true language grounding = weak
```

Bu VLA evaluation için çok önemli bir problemdir.

---

# 37. Distribution Shift

Training ve test distribution'ı aynı değilse distribution shift oluşur.

Örneğin:

```text
training:
bright room

test:
dark room
```

veya:

```text
training:
"pick up the red mug"

test:
"could you grab the crimson cup?"
```

Model bunu başarabiliyor mu?

Robustness çalışmalarının temel sorularından biridir.

---

# 38. Domain Randomization

Simulation'da environment özellikleri randomize edilir.

Örneğin:

- lighting
- texture
- object position
- camera angle
- friction

Amaç:

Modelin tek environment configuration'ına aşırı uyum sağlamasını önlemek.

---

# 39. Sim-to-Real

Model simulation'da eğitilir.

Sonra gerçek robota aktarılır.

Problem:

```text
simulation != reality
```

Bu fark yüzünden model gerçek dünyada kötü çalışabilir.

Bu problem:

**sim-to-real transfer**

olarak adlandırılır.

---

# 40. Real-to-Sim

Tersi yaklaşım da vardır.

Gerçek dünyadaki robot veya environment'ın dijital kopyası oluşturulur.

Bu özellikle digital twin yaklaşımında önemlidir.

---

# 41. World Model

World model, environment'ın nasıl değişeceğini tahmin etmeye çalışan modeldir.

Örneğin:

```text
current observation
+
candidate action
    |
    v
predicted next observation
```

Robot planlama için world model kullanabilir.

VLA ile world model aynı şey değildir.

Ama gelecekte bu iki yaklaşım giderek daha fazla birleşebilir.

---

# 42. Diffusion Policy

Diffusion Policy robot manipulation için önemli policy ailelerinden biridir.

VLA değildir.

Genellikle continuous action sequence üretmek için diffusion model yaklaşımını kullanır.

Basit karşılaştırma:

```text
VLA
vision + language -> action

Diffusion Policy
observation -> continuous action trajectory
```

Bazı modern sistemler language ile diffusion policy yaklaşımını da birleştirebilir.

---

# 43. Transformer Robot Policy

Birçok modern robot policy Transformer mimarisi kullanır.

Çünkü robot task'leri zaman serisi, multimodal input ve uzun context içerir.

Örneğin:

```text
image tokens
language tokens
robot state tokens
previous action tokens
```

aynı model içinde işlenebilir.

---

# 44. Action Chunking

Model her timestep için tek action üretmek yerine birkaç action'ı birden üretebilir.

Örneğin:

```text
input:
current observation

output:
a_t
a_t+1
a_t+2
...
a_t+9
```

Bu yaklaşım robot control'de daha stabil davranış sağlayabilir.

ACT gibi modellerde bu fikir önemlidir.

---

# 45. Closed-Loop ve Open-Loop

## Open-loop

Model bir action sequence üretir ve robot bunu yeni observation almadan uygular.

```text
observe once
   |
predict 10 actions
   |
execute all
```

## Closed-loop

Robot sürekli yeni observation alır.

```text
observe
act
observe
act
observe
act
```

Gerçek dünya için closed-loop control genellikle daha güvenlidir.

---

# 46. Robot Learning'deki büyük ayrım

Bir paper gördüğünde önce şunu sor:

```text
Bu çalışma neyi değiştiriyor?
```

Seçenekler genellikle:

```text
MODEL
DATA
TRAINING
ENVIRONMENT
BENCHMARK
HARDWARE
CONTROL
EVALUATION
```

Bu soruyu sormak terim karmaşasını ciddi şekilde azaltır.

---

# 47. Terimleri doğru kategoriye yerleştirme

## Model / Policy

```text
OpenVLA
π0
RT-2
Octo
Diffusion Policy
ACT
```

## Training / Fine-tuning Method

```text
OpenVLA-OFT
behavior cloning
imitation learning
RL
```

## Benchmark / Task Suite

```text
LIBERO
LIBERO-Plus
CALVIN
RLBench
```

## Simulator / Physics

```text
MuJoCo
Isaac Sim
PyBullet
Gazebo
```

## Robot Learning Framework / Tooling

```text
LeRobot
robomimic
```

## Robot Middleware

```text
ROS
ROS2
```

## Hardware

```text
Franka
UR5
ALOHA
Microduck
```

---

# 48. Yeni bir paper nasıl okunmalı?

Bir VLA paper'ını ilk kez okurken architecture diagram ile başlamamak daha iyi olabilir.

Önce şu soruları cevapla.

## 1. Task

Robot ne yapıyor?

```text
pick?
place?
open?
long-horizon manipulation?
navigation?
```

## 2. Environment

```text
simulation?
real robot?
both?
```

## 3. Data

Model ne ile öğreniyor?

```text
human demonstrations?
robot trajectories?
internet data?
simulation data?
```

## 4. Input

Model ne görüyor?

```text
RGB?
depth?
language?
robot state?
history?
```

## 5. Output

Model ne üretiyor?

```text
joint action?
end-effector pose?
action chunk?
high-level plan?
```

## 6. Model

Hangi policy kullanılıyor?

```text
VLA?
diffusion?
transformer?
world model?
```

## 7. Training

Nasıl optimize edilmiş?

```text
behavior cloning?
fine-tuning?
RL?
LoRA?
full fine-tuning?
```

## 8. Benchmark

Nerede test edilmiş?

```text
LIBERO?
CALVIN?
real robot?
```

## 9. Generalization Claim

Yazar neyin generalize olduğunu iddia ediyor?

```text
new objects?
new tasks?
new language?
new environments?
```

## 10. Ablation

Yazar hangi input'u çıkarınca ne oluyor?

Bu özellikle VLA için çok önemlidir.

---

# 49. VLA paper'larında özellikle dikkat edilmesi gereken şey

Başarı metric'i tek başına yeterli değildir.

Örneğin:

```text
Model A
LIBERO success = 98%
```

Bu bize şunu söylemez:

```text
language gerçekten kullanılıyor mu?
```

Bu yüzden güçlü evaluation şunları test etmelidir:

```text
wrong language
blank language
paraphrased language
different object layout
different camera
different texture
distractor objects
new environment
```

---

# 50. OpenVLA-OFT neden karşına çıktı?

Sen OpenVLA + LIBERO çizgisinde araştırma yaptığında literatür doğal olarak seni OpenVLA-OFT'ye götürür.

Çünkü:

```text
OpenVLA
   |
fine-tune on robot task
   |
LIBERO evaluation
```

konusunda OpenVLA-OFT önemli bir referanstır.

Yani OFT paper'ının karşına çıkması yanlış değildir.

Ama kritik nokta şudur:

```text
OpenVLA-OFT result
```

otomatik olarak:

```text
base OpenVLA behavior
```

demek değildir.

Fine-tuning sonrası model language'e, vision'a veya environment shortcut'larına farklı ağırlık verebilir.

---

# 51. LIBERO-Plus bulgusu neden ilginç?

Normal beklenti:

```text
Vision + Language -> Action
```

Eğer language input kaldırıldığında performans aynı kalıyorsa şu soru doğar:

```text
Model gerçekten language kullanıyor mu?
```

Olası açıklama:

```text
Environment görüntüsü
        |
        v
task identity already obvious
        |
        v
language becomes redundant
```

Bu VLA benchmark tasarımında önemli bir problemdir.

---

# 52. Bu alan için doğru öğrenme sırası

Bir başlangıç sırası şöyle olabilir.

## Aşama 1

Şunları öğren:

```text
observation
action
policy
trajectory
episode
behavior cloning
imitation learning
```

## Aşama 2

Robot manipulation:

```text
gripper
end-effector
joint
pose
control
```

## Aşama 3

Simulation:

```text
MuJoCo
environment
task
reset
step
reward
```

## Aşama 4

Modern robot policies:

```text
ACT
Diffusion Policy
Octo
OpenVLA
π0
```

## Aşama 5

Benchmarks:

```text
LIBERO
CALVIN
RLBench
```

## Aşama 6

Robustness ve generalization:

```text
language grounding
visual perturbation
distribution shift
shortcut learning
```

---

# 53. Pratik başlangıç stratejisi

Gerçek robot olmadan başlanabilir.

İlk hedef:

```text
existing benchmark
+
existing policy
+
existing checkpoint
```

çalıştırmak.

Örneğin:

```text
LIBERO
+
existing OpenVLA / policy implementation
```

İlk amaç yeni model geliştirmek değil.

Şunu anlamak:

```text
dataset nasıl geliyor?
environment nasıl resetleniyor?
observation nasıl modele gidiyor?
action nasıl environment'a uygulanıyor?
success nasıl hesaplanıyor?
```

Bu pipeline bir kez çalışınca alan çok daha anlaşılır hale gelir.

---

# 54. Bir VLA deneyinin tam akışı

```text
TASK
"pick up the red cup"

        |
        v

ENVIRONMENT
robot + table + objects

        |
        v

OBSERVATION
RGB + robot state + language

        |
        v

POLICY
OpenVLA / π0 / başka model

        |
        v

ACTION

        |
        v

ENVIRONMENT STEP

        |
        v

NEW OBSERVATION

        |
        v

repeat

        |
        v

SUCCESS / FAILURE
```

---

# 55. Simülasyon ve gerçek robot bağlantısı

```text
                    POLICY
                       |
           +-----------+-----------+
           |                       |
           v                       v
      SIMULATION               REAL ROBOT
       MuJoCo                   Franka
      Isaac Sim                  UR5
      PyBullet                  Microduck
```

Aynı policy teorik olarak hem simulation hem gerçek robot üzerinde çalışabilir.

Ama action space, sensors ve robot morphology uyumlu olmalıdır.

---

# 56. ROS nerede devreye girer?

Gerçek robotta tipik yapı:

```text
CAMERA
  |
  v
ROS IMAGE TOPIC
  |
  v
VLA MODEL
  |
  v
ACTION
  |
  v
ROS CONTROLLER
  |
  v
ROBOT
```

ROS burada model değildir.

Bileşenlerin birbirine bağlanmasını sağlar.

---

# 57. LeRobot nerede devreye girer?

Tipik akış:

```text
REAL ROBOT
   |
teleoperation
   |
   v
LeRobot Dataset
   |
   v
Policy Training
   |
   v
Evaluation
   |
   v
Deployment
```

LeRobot tüm workflow'u kolaylaştıran bir tool ecosystem olarak düşünülebilir.

---

# 58. En kritik kavramsal ayrım

Aşağıdaki tabloyu zihninde tutmak yeterlidir.

| Soru | Kavram |
|---|---|
| Robot ne yapacak? | Task |
| Robot ne görüyor? | Observation |
| Kararı kim veriyor? | Policy |
| Robot ne uyguluyor? | Action |
| Model nasıl öğreniyor? | Training method |
| Nerede çalışıyor? | Environment |
| Fizik nerede simüle ediliyor? | Simulator |
| Model nasıl sınanıyor? | Benchmark |
| Gerçek robot parçaları nasıl konuşuyor? | ROS |
| Robot learning workflow'unu ne kolaylaştırıyor? | LeRobot |
| Fiziksel makine nedir? | Hardware |

---

# 59. Tek cümlede alan

Robot learning:

> Gözlemlerden robot action üreten policy'lerin veriden öğrenilmesini inceler.

VLA:

> Vision ve language bilgisini kullanarak robot action üreten modern policy ailesidir.

Simulator:

> Robotun fiziksel dünyasını bilgisayarda taklit eder.

Benchmark:

> Modelin belirli görevlerde ne kadar iyi çalıştığını ölçer.

ROS:

> Gerçek robot sistemindeki yazılım bileşenlerini birbirine bağlar.

LeRobot:

> Robot learning için veri toplama, training ve deployment araçları sağlar.

---

# 60. Bundan sonra yeni bir terim gördüğünde

İlk soru:

```text
Bu şey nedir?
```

Aşağıdaki kategorilerden birine koy:

```text
MODEL
TRAINING METHOD
DATASET
BENCHMARK
SIMULATOR
FRAMEWORK
MIDDLEWARE
HARDWARE
CONTROL METHOD
EVALUATION METHOD
```

Bu sınıflandırmayı yaptıktan sonra terimin anlamı genellikle çok daha kolay oturur.

---

# Son zihinsel harita

```text
EMBODIED AI
|
+-- ROBOT LEARNING
    |
    +-- DATA
    |   +-- demonstrations
    |   +-- trajectories
    |
    +-- LEARNING
    |   +-- behavior cloning
    |   +-- imitation learning
    |   +-- reinforcement learning
    |
    +-- POLICY
    |   |
    |   +-- ACT
    |   +-- Diffusion Policy
    |   +-- VLA
    |       |
    |       +-- OpenVLA
    |       +-- π0
    |       +-- Octo
    |       +-- RT-2
    |
    +-- ENVIRONMENT
    |   |
    |   +-- Simulation
    |   |   +-- MuJoCo
    |   |   +-- Isaac Sim
    |   |   +-- PyBullet
    |   |
    |   +-- Real World
    |       +-- Franka
    |       +-- UR5
    |       +-- ALOHA
    |       +-- Microduck
    |
    +-- BENCHMARK
    |   |
    |   +-- LIBERO
    |   +-- LIBERO-Plus
    |   +-- CALVIN
    |   +-- RLBench
    |
    +-- TOOLING
        |
        +-- LeRobot
        +-- robomimic
        +-- ROS / ROS2
```

Bu harita model isimlerinden bağımsız olarak alanı okumak için kullanılabilir.

Yeni bir paper geldiğinde önce paper'ın bu haritada hangi kutuyu değiştirdiğini bulmak gerekir.

Sonra teknik detaylara geçmek çok daha kolay olur.
