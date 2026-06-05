class Model {
public:
    // Model(std::string);

    virtual void train(const DataSet& dataset) = 0;
    virtual long getsizeof() const = 0;
    // Minimal deployed state in bytes — one yardstick across model families: the
    // learned table/seen-set needed at inference. Default = in-memory getsizeof();
    // Wisard/ClusWisard/BloomWisard override with their deployable representation.
    virtual long deployedSizeBytes() const { return getsizeof(); }
    virtual std::string json(std::string filename) const = 0;

    std::string json() const {
        return json("");
    }
};