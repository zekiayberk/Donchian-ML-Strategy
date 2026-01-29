
# Donchian Breakout + ATR Trailing Stop Stratejisi

## Strateji Mantığı
Bu strateji, klasik bir "Trend Following" (Trend Takibi) yaklaşımıdır. Piyasaların zamanın belli bir kısmında güçlü trendler oluşturduğu varsayımına dayanır. Amacı, bu trendleri mümkün olduğunca erken yakalamak ve trend bitene kadar pozisyonda kalmaktır.

### 1. Giriş Sinyali (Donchian Channel Breakout)
Donchian Kanalı, son N periyodun en yüksek ve en düşük fiyatlarından oluşur.
- **Kural**: Fiyat, son 20 günün (veya barın) en yükseğini yukarı kırarsa **LONG** (Alış) yapılır.
- **Kural**: Fiyat, son 20 günün en düşüğünü aşağı kırarsa **SHORT** (Satış) yapılır.
- **Mantık**: Fiyatın yeni bir tepe yapması, alıcıların güçlü olduğunun ve trendin yukarı yönlü olabileceğinin işaretidir.

### 2. Çıkış Sinyali ve Risk Yönetimi (ATR Trailing Stop)
Trend takibi stratejilerinde en önemli kısım "nerede çıkılacağıdır". Kar hedefleri (Take Profit) kullanılmaz, bunun yerine karı koruyan iz süren stoplar (Trailing Stop) kullanılır.
- **ATR (Average True Range)**: Piyasanın volatilitesini ölçer.
- **Stop Loss**: Giriş fiyatından `k * ATR` kadar uzakta başlar.
- **Trailing Stop**: 
  - Long pozisyon için: Fiyat yükseldikçe stop seviyesi de yukarı çekilir. Stop seviyesi asla aşağı düşürülmez.
  - Formül: `New Stop = Max(Previous Stop, High - k * ATR)`
- **Mantık**: ATR kullanımı, stop mesafesinin piyasa oynaklığına göre dinamik olmasını sağlar. Çok oynak piyasada geniş stop (erken stop olmamak için), sakin piyasada dar stop kullanılır.

## Parametreler
- **Donchian Period (N)**: 20 (Daha kısa = daha agresif, daha sık sinyal, daha çok false signal).
- **ATR Period**: 14 (Standart volatilite ölçümü).
- **Stop Multiplier (k)**: 2.5 (Stop mesafesi çarpanı).
- **Timeframe**: 1 Saat (1h) veya 4 Saat (4h) önerilir.

## Risk Yönetimi Kuralları
1. **Sermaye Riski**: Her işlemde toplam sermayenin en fazla %0.5'i riske edilir.
   - Örnek: 10.000$ sermaye -> Max kayıp 50$.
   - Pozisyon Büyüklüğü = 50$ / (Giriş Fiyatı - Stop Fiyatı).
2. **Kaldıraç**: Sadece marjin gereksinimini karşılamak için kullanılır, risk miktarını artırmak için değil.

## Güçlü ve Zayıf Yönler
- **Güçlü Yönler**: Büyük rallileri ve çöküşleri yakalar (Örn: BTC 20k -> 60k). Duygusal kararları ortadan kaldırır.
- **Zayıf Yönler**: Yatay (testere/chop) piyasalarda sık sık stop olur ve küçük zararlar yazar. Başarı oranı (Win Rate) genellikle düşüktür (%35-45), ancak Kazanç/Kayıp oranı yüksektir.

## Hangi Koşullarda Çalışmaz?
- Düşük volatilite ve yatay piyasa koşulları.
- "Mean Reversion" (Ortalamaya dönüş) karakteri gösteren piyasalar.

## Notlar
- Futures işleminde "Funding Rate" maliyeti uzun süreli taşımalarda karı eritebilir.
- Kayma (Slippage), breakout anlarında yüksek olabilir, backtestlerde bu mutlaka hesaba katılmalıdır (bu projede `slippage_bps` parametresi ile eklendi).
