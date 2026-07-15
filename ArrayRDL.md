addrmap top {
    name = "top";

    mem {
        name = "register_bank";
        mementries = 231;   // number of elements
        memwidth = 64;      // must match your 8-byte stride, NOT the 32-bit field width
        reg {
            name = "Register_name";
            regwidth = 32;
            field {
                sw = rw;
                hw = rw;
                reset = 0x0;
            } field1[31:0];
        } register_name[231] @ 0x0 += 0x8;
    } external register_bank @ 0x0;
};