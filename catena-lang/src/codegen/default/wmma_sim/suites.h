// Directed BF16 suites adapted from the standalone validator.
    void golden_cases() {
        const uint16_t one = encode16(false, 0), neg_one = one ^ 0x8000;
        const uint16_t small = encode16(false, -12);
        auto check = [&](uint16_t a0, uint16_t b0, uint16_t a1, uint16_t b1, uint32_t c, uint32_t expected) {
            const uint32_t actual = simulate_dot2(a0,b0,a1,b1,c);
            if (actual != expected) throw std::runtime_error("CPU arithmetic golden case failed");
            ++golden_checks;
        };
        check(one,one,0,0,0xbf800000,as_bits(0x1p-23f));
        check(neg_one,one,0,0,0x3f800000,as_bits(-0x1p-24f));
        check(one,one,neg_one,one,0,as_bits(-0x1p-24f));
        check(one,one,small,small,0xbf800000,as_bits(0x1.8p-23f));
        check(0x8000,one,one,one,0,0x3f7fffff);
        check(0,0,0,0,0x80000001,0);
        check(0,0,0,0,0x80800000,0x80800000);
        check(0,0,0,0,0x80000000,0);
        const uint16_t inf = 0x7f80;
        check(inf,0,0,0,0,INVALID_NAN);
        check(inf,0,0,0,0x7fc12345,INVALID_NAN);
        check(uint16_t(inf|1),one,inf,0,0xffc12345,OPERAND_NAN);
        check(inf,0,uint16_t(inf|1),one,0,OPERAND_NAN);
        check(0,0,0,0,0x7f800001,0x7fc00001);
        check(0,0,0,0,0xff812345,0xffc12345);
        check(inf,one,0,0,0x7f812345,0x7fc12345);
        check(inf,one,uint16_t(inf^0x8000),one,0x7fc12345,INVALID_NAN);
        check(encode16(false,-100),encode16(false,-51),0x8000,one,0x80800000,0);
        check(encode16(false,-100),encode16(false,-52),0x8000,one,0x80800000,0x80800000);
        check(0x8001,one,one,one,0,0x3f7fffff);
        std::cout << "Golden dot2 cases: " << golden_checks << " passed\n";
    }
    template<class Generator>
    void suite(const std::string& label, uint64_t total, uint64_t seed, Generator generator, uint64_t cases = 0) {
        Result result; result.suite = label; result.seed = seed; result.diagonal_cases = cases;
        for (uint64_t offset = 0; offset < total;) {
            const size_t n = size_t(std::min<uint64_t>(opt.batch, total - offset));
            std::vector<Tile> batch; batch.reserve(n);
            for (size_t i = 0; i < n; ++i) batch.push_back(generator(offset + i));
            compare(result, batch); offset += n;
        }
        finish(result);
    }
    void mapping() {
        // One-hot B selects each k independently, for every output row/column.
        // The selected integer values are exactly representable in BF16.
        suite("lane_mapping", 16, 0, [&](uint64_t k) {
            Tile t;
            for (int m = 0; m < 16; ++m) for (int j = 0; j < 16; ++j) t.a[m*16+j] = encode16(false,(m+j)%4);
            for (int n = 0; n < 16; ++n) t.bt[n*16+(k+n)%16] = encode16(false,0);
            for (int m = 0; m < 16; ++m) for (int n = 0; n < 16; ++n) {
                const uint32_t expected = encode32(false,(m+int((k+n)%16))%4);
                const uint32_t actual = simulate_output(t.a.data()+m*16,t.bt.data()+n*16,0);
                if (actual != expected) throw std::runtime_error("independent lane-mapping golden case failed");
                ++golden_checks;
            }
            return t;
        });
    }
    void special() {
        const uint16_t inf = 0x7f80;
        const uint16_t max = inf - 1, tiny = 0x0080;
        const uint16_t one = encode16(false,0);
        const std::array<uint16_t,13> values = {0,0x8000,one,uint16_t(one^0x8000),inf,uint16_t(inf^0x8000),
            uint16_t(inf|1),max,uint16_t(max^0x8000),tiny,uint16_t(tiny^0x8000),1,0x8001};
        const std::array<uint32_t,17> cv = {0,0x80000000,0x3f800000,0xbf800000,INF,INF|0x80000000,QNAN,
            0x7f7fffff,0xff7fffff,0x00800000,0x80800000,1,0x80000001,0x007fffff,0x807fffff,0x00800001,0x80800001};
        const uint64_t cases = 13ull*13*13*13*17;
        suite("special_values",(cases+15)/16,0,[&](uint64_t tile_index) {
            Tile t;
            for (int lane=0;lane<16;++lane) {
                uint64_t id=tile_index*16+lane; if(id>=cases) break;
                t.c[lane*16+lane]=cv[id%17]; id/=17;
                const auto b1=values[id%13]; id/=13;
                const auto a1=values[id%13]; id/=13;
                const auto b0=values[id%13]; id/=13;
                const auto a0=values[id%13];
                t.a[lane*16]=a0; t.a[lane*16+1]=a1; t.bt[lane*16]=b0; t.bt[lane*16+1]=b1;
            }
            return t;
        }, cases);
    }
    void boundary() {
        const int low=-100, high=-50;
        const uint64_t length=high-low+1, cases=length*length*8*17;
        suite("normal_boundary",(cases+15)/16,0,[&](uint64_t tile_index) {
            Tile t;
            for(int lane=0;lane<16;++lane) {
                uint64_t id=tile_index*16+lane; if(id>=cases)break;
                const int offset=int(id%17)-8; id/=17;
                const bool cs=id%2; id/=2; const bool bs=id%2; id/=2; const bool as=id%2; id/=2;
                const int eb=low+int(id%length); id/=length; const int ea=low+int(id%length);
                t.a[lane*16]=encode16(as,ea); t.bt[lane*16]=encode16(bs,eb);
                t.a[lane*16+1]=cs?0x8000:0; t.bt[lane*16+1]=encode16(false,0);
                t.c[lane*16+lane]=uint32_t(0x00800000+offset)|(cs?0x80000000u:0);
            }
            return t;
        },cases);
    }
    void all_encodings() {
        // Every 16-bit input encoding appears in each multiplicand role, with
        // cancellation, signed zero, infinities, and varied accumulators.
        const uint16_t one=encode16(false,0);
        for(int role=0;role<4;++role) suite("all_encodings_role_"+std::to_string(role),4096,0,[&](uint64_t tile_index) {
            Tile t;
            for(int lane=0;lane<16;++lane) {
                const uint16_t x=uint16_t(tile_index*16+lane);
                uint16_t v[4]={one,one,uint16_t(one^0x8000),one}; v[role]=x;
                t.a[lane*16]=v[0];t.bt[lane*16]=v[1];t.a[lane*16+1]=v[2];t.bt[lane*16+1]=v[3];
                const uint32_t cs[]={0,0x80000000,0x3f800000,0xbf800000,0x00800000,0x80800000,INF,QNAN};
                t.c[lane*16+lane]=cs[x%8];
            }
            return t;
        },65536);
    }
