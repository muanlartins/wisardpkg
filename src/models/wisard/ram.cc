
class RAM{
public:
  RAM(){}
  RAM(nl::json c){
    ignoreZero = c["ignoreZero"];
    base=c["base"];
    addresses = c["addresses"].get<std::vector<int>>();
    useLargeAddr = (base == 2 && (int)addresses.size() > 64);

    if(!useLargeAddr){
      checkLimitAddressSize(addresses.size(), base);
      RAMDataHandle handle(c["data"].get<std::string>());
      positions = handle.get(0);
    }
    // Large-address deserialization not supported (not needed for experiments)
  }
  RAM(const int addressSize, const int entrySize, const bool ignoreZero=false, int base=2): ignoreZero(ignoreZero), base(base){
    useLargeAddr = (base == 2 && addressSize > 64);
    if(!useLargeAddr){
      checkLimitAddressSize(addressSize, base);
    }
    addresses = std::vector<int>(addressSize);
    generateRandomAddresses(entrySize);
  }
  RAM(const std::vector<int> indexes, const bool ignoreZero=false, int base=2): addresses(indexes), ignoreZero(ignoreZero), base(base){
    useLargeAddr = (base == 2 && (int)indexes.size() > 64);
    if(!useLargeAddr){
      checkLimitAddressSize(indexes.size(), base);
    }
  }

  int getVote(const BinInput& image) const {
    if(useLargeAddr){
      large_addr_t key = getLargeIndex(image);
      if(ignoreZero){
        bool allZero = true;
        for(char c : key) if(c != 0){ allZero = false; break; }
        if(allZero) return 0;
      }
      auto it = largePositions.find(key);
      return (it != largePositions.end()) ? it->second : 0;
    }
    addr_t index = getIndex(image);
    if(ignoreZero && index == 0)
      return 0;
    auto it = positions.find(index);
    if(it == positions.end()){
      return 0;
    }
    else{
      return it->second;
    }
  }

  void train(const BinInput& image){
    if(useLargeAddr){
      large_addr_t key = getLargeIndex(image);
      auto it = largePositions.find(key);
      if(it == largePositions.end()){
        largePositions.insert(it, std::pair<large_addr_t,content_t>(key, 1));
      } else {
        it->second++;
      }
      return;
    }
    addr_t index = getIndex(image);
    auto it = positions.find(index);
    if(it == positions.end()){
      positions.insert(it,std::pair<addr_t,content_t>(index, 1));
    }
    else{
      it->second++;
    }
  }

  void untrain(const BinInput& image){
    if(useLargeAddr){
      large_addr_t key = getLargeIndex(image);
      auto it = largePositions.find(key);
      if(it != largePositions.end()){
        it->second--;
      }
      return;
    }
    addr_t index = getIndex(image);
    auto it = positions.find(index);
    if(it != positions.end()){
      it->second--;
    }
  }

  void reset() {
    positions.clear();
    largePositions.clear();
  }

  std::vector<std::vector<int>> getMentalImage() {
    std::vector<std::vector<int>> mentalPiece(addresses.size());
    for(unsigned int i=0; i<mentalPiece.size(); i++){
      mentalPiece[i].resize(2);
      mentalPiece[i][0] = addresses[i];
      mentalPiece[i][1] = 0;
    }

    if(useLargeAddr){
      for(auto j=largePositions.begin(); j!=largePositions.end(); ++j){
        const large_addr_t& key = j->first;
        for(unsigned int i=0; i<mentalPiece.size(); i++){
          int byteIdx = i / 8;
          int bitIdx = i % 8;
          if(byteIdx < (int)key.size() && (key[byteIdx] & (1 << bitIdx))){
            mentalPiece[i][1] += j->second;
          }
        }
      }
      return mentalPiece;
    }

    for(auto j=positions.begin(); j!=positions.end(); ++j){
      if(j->first == 0) continue;
      const std::vector<int> address = convertToBase(j->first);
      for(unsigned int i=0; i<mentalPiece.size(); i++){
        if(mentalPiece[i].size() == 0){
          mentalPiece[i].resize(2);
          mentalPiece[i][0] = addresses[i];
          mentalPiece[i][1] = 0;
        }
        if(address[i] > 0){
          mentalPiece[i][1] += j->second;
        }
      }
    }
    return mentalPiece;
  }

  nl::json getJson() const{
    nl::json config = {
      {"ignoreZero", ignoreZero},
      {"base", base}
    };
    return config;
  }

  std::string getData() const {
    if(useLargeAddr){
      // Large-address serialization: return empty (not needed for experiments)
      ram_t empty_ram;
      RAMDataHandle handle(empty_ram);
      return handle.data(0);
    }
    RAMDataHandle handle(positions);
    return handle.data(0);
  }

  void setMapping(std::vector<std::vector<int>>& mapping, int i) const {
    int size = addresses.size();
    mapping[i].resize(size);
    for(int j=0; j<size; j++) {
      mapping[i][j] = addresses[j];
    }
  }

  std::vector<int> getMapping() const{
    return addresses;
  }

  int getAddressSize(){
    return addresses.size();
  }

  int getTupleSize() const{
    return addresses.size();
  }

  long getsizeof() const{
    long size = sizeof(RAM);
    size += addresses.size()*sizeof(int);
    if(useLargeAddr){
      int keyBytes = ((int)addresses.size() + 7) / 8;
      size += largePositions.size()*(keyBytes+sizeof(content_t));
    } else {
      size += positions.size()*(sizeof(addr_t)+sizeof(content_t));
    }
    return size;
  }

  long getNumEntries() const{
    return useLargeAddr ? largePositions.size() : positions.size();
  }

  // Minimal deployed footprint of this RAM: the set of distinct seen addresses,
  // each bit-packed as ceil(tupleSize/8) bytes (lossless, no false positives,
  // unlike BloomWisard's fixed filter). The deployable analogue of the
  // BTHOWeN/ULEEN/Bloom bit-table, on one comparable yardstick.
  long deployedSizeBytes() const{
    long bytesPerAddr = ((long)addresses.size() + 7) / 8;
    return getNumEntries() * bytesPerAddr;
  }

  ~RAM(){
    addresses.clear();
    positions.clear();
    largePositions.clear();
  }

protected:
  addr_t getIndex(const BinInput& image) const{
    addr_t index = 0;
    addr_t p = 1;
    for(unsigned int i=0; i<addresses.size(); i++){
      int bin = image[addresses[i]];
      checkPos(bin);
      index += bin*p;
      p *= base;
    }
    return index;
  }

  large_addr_t getLargeIndex(const BinInput& image) const{
    int n = addresses.size();
    int nbytes = (n + 7) / 8;
    large_addr_t key(nbytes, '\0');
    for(int i = 0; i < n; i++){
      if(image[addresses[i]]){
        key[i / 8] |= (char)(1 << (i % 8));
      }
    }
    return key;
  }


private:
  std::vector<int> addresses;
  ram_t positions;
  large_ram_t largePositions;
  bool useLargeAddr = false;
  bool ignoreZero;
  int base;

  const std::vector<int> convertToBase(const int number) const{
    std::vector<int> numberConverted(addresses.size());
    int baseNumber = number;
    for(unsigned int i=0; i<numberConverted.size(); i++){
      numberConverted[i] = baseNumber % base;
      baseNumber /= base;
    }
    return numberConverted;
  }

  void checkLimitAddressSize(int addressSize, int basein){
    const addr_t limit = -1;
    if((basein == 2 && addressSize > 64) ||
       (basein != 2 && (addr_t)ipow(basein,addressSize) > limit)){
      throw Exception("The base power to addressSize passed the limit of 2^64!");
    }
  }

  void checkPos(const int code) const{
    if(code >= base){
      throw Exception("The input data has a value bigger than base of addresing!");
    }
  }

  void generateRandomAddresses(int entrySize){
    for(unsigned int i=0; i<addresses.size(); i++){
      addresses[i] = randint(0, entrySize-1);
    }
  }
};
