class RandomMapping : public MappingGenerator{
public:
    RandomMapping(const bool monoMapping=false, const bool completeAddressing=true) {
        init(std::vector<int>(), 0, monoMapping, completeAddressing);
    }

    RandomMapping(const std::vector<int> indexes, const unsigned int tupleSize, const bool monoMapping=false, const bool completeAddressing=true) {
        init(indexes, tupleSize, monoMapping, completeAddressing);
    }

    RandomMapping(const unsigned int entrySize, const unsigned int tupleSize, const bool monoMapping=false, const bool completeAddressing=true) {
        init(entrySizeToIndexes(entrySize), tupleSize, monoMapping, completeAddressing);
    }

    RandomMapping(nl::json config){
        nl::json value;

        value = config["indexes"];
        indexes = value.get<std::vector<int>>();

        value = config["mapping"];
        mapping = value.get<std::map<std::string, std::vector<std::vector<int>>>>();

        value = config["tupleSize"];
        tupleSize = value.get<unsigned int>();

        value = config["completeAddressing"];
        completeAddressing = value.get<bool>();

        value = config["monoMapping"];
        monoMapping = value.get<bool>();
    }

    std::vector<std::vector<int>> getMapping(const std::string label){
        checkEntrySize(indexes.size());
        checkTupleSize(tupleSize);

        auto it = mapping.find(label);
        if (it != mapping.end()){
            return it->second;
        } else if (monoMapping && mapping.size() > 0){
            return mapping.begin()->second;
        }

        mapping[label] = createMapping(tupleSize, indexes, completeAddressing);
        return mapping[label];
    }

    MappingGenerator* clone() const{
        return new RandomMapping(indexes, tupleSize, monoMapping, completeAddressing);
    }

    std::string json() const{
        nl::json config = {
            {"indexes", indexes},
            {"mapping", mapping},
            {"tupleSize", tupleSize},
            {"completeAddressing", completeAddressing},
            {"monoMapping", monoMapping}
        };
        return config.dump();
    }

    std::string className() const{
        return "RandomMapping";
    }

protected:
    std::vector<std::vector<int>> createMapping(const unsigned int tupleSize, const std::vector<int>& indexes, const bool completeAddressing) const{
        std::vector<int> mappingIndexes = indexes;

        std::random_device rd;
        std::mt19937 mt(rd());

        if (multiResolution) {
            // Multi-Resolution: deterministic size distribution, random bit assignment.
            // Same number of RAMs as standard. Sizes linearly spaced from min to max,
            // then scaled so total = entrySize. Shuffled before assignment.
            unsigned int entrySize = mappingIndexes.size();
            unsigned int numRAMs = entrySize / tupleSize;
            if (numRAMs == 0) numRAMs = 1;
            if (entrySize % tupleSize != 0 && completeAddressing) numRAMs++;

            unsigned int minSize = std::max((unsigned int)2, tupleSize / 2);
            unsigned int maxSize = tupleSize * 3 / 2;
            if (maxSize < minSize) maxSize = minSize;

            // Generate linearly spaced sizes
            std::vector<unsigned int> sizes(numRAMs);
            if (numRAMs == 1) {
                sizes[0] = entrySize;
            } else {
                for (unsigned int i = 0; i < numRAMs; i++) {
                    double frac = (double)i / (double)(numRAMs - 1);
                    sizes[i] = (unsigned int)(minSize + frac * (maxSize - minSize) + 0.5);
                }
            }

            // Scale to exactly cover entrySize bits
            unsigned int currentTotal = 0;
            for (unsigned int i = 0; i < numRAMs; i++) currentTotal += sizes[i];

            // Distribute surplus/deficit across RAMs
            int delta = (int)entrySize - (int)currentTotal;
            while (delta > 0) {
                for (unsigned int i = numRAMs; i > 0 && delta > 0; i--) {
                    sizes[i-1]++; delta--;
                }
            }
            while (delta < 0) {
                for (unsigned int i = 0; i < numRAMs && delta < 0; i++) {
                    if (sizes[i] > 2) { sizes[i]--; delta++; }
                }
            }

            // Shuffle sizes so position doesn't determine size
            std::shuffle(sizes.begin(), sizes.end(), mt);

            // Shuffle input bits then assign variable-size chunks
            std::shuffle(mappingIndexes.begin(), mappingIndexes.end(), mt);

            std::vector<std::vector<int>> result;
            unsigned int pos = 0;
            for (unsigned int i = 0; i < numRAMs && pos < entrySize; i++) {
                unsigned int sz = sizes[i];
                if (pos + sz > entrySize) sz = entrySize - pos;
                result.push_back(std::vector<int>(
                    mappingIndexes.begin() + pos,
                    mappingIndexes.begin() + pos + sz));
                pos += sz;
            }
            return result;
        }

        // Standard: uniform-size chunks
        if (completeAddressing){
            mappingIndexes = completeMapping(tupleSize, mappingIndexes);
        }

        std::shuffle(mappingIndexes.begin(), mappingIndexes.end(), mt);

        unsigned int numberOfRAMS = mappingIndexes.size() / tupleSize;
        std::vector<std::vector<int>> mapping(numberOfRAMS);

        for(unsigned int i = 0; i < numberOfRAMS; i++){
            mapping[i] = std::vector<int>(mappingIndexes.begin() + (i*tupleSize), mappingIndexes.begin() + ((i+1)*tupleSize));
        }

        return mapping;
    }

    void init(const std::vector<int> indexes, const unsigned int tupleSize, const bool monoMapping, const bool completeAddressing){
        this->indexes = indexes;
        this->tupleSize = tupleSize;
        this->monoMapping = monoMapping;
        this->completeAddressing = completeAddressing;
        this->multiResolution = false;
    }
};
