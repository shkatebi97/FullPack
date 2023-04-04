#include <iostream>
#include <bitset>
int main(){
    int as[] = { -1,  0,  0 };
    int ws[] = {  1,  1,  1 };
    std::cout << "a = { " 
              << as[0] << " (0b" << std::bitset<6>((as[0] & 0x3F)) << "), "
              << as[1] << " (0b" << std::bitset<6>((as[1] & 0x3F)) << "), "
              << as[2] << " (0b" << std::bitset<6>((as[2] & 0x3F)) << ") }";
    int16_t a = 0 | ((as[0] & 0x3F) << 0) | ((as[1] & 0x3F) << 6) | ((as[2] & 0x3F) << 12);
    std::cout << " -> 0b" << std::bitset<16>(a) << std::endl;
    std::cout << "w = { " 
              << ws[0] << " (0b" << std::bitset<6>((ws[0] & 0x3F)) << "), "
              << ws[1] << " (0b" << std::bitset<6>((ws[1] & 0x3F)) << "), "
              << ws[2] << " (0b" << std::bitset<6>((ws[2] & 0x3F)) << ") }";
    int16_t w = 0 | ((ws[2] & 0x3F) << 0) | ((ws[1] & 0x3F) << 6) | ((ws[0] & 0x3F) << 12);
    std::cout << " -> 0b" << std::bitset<16>(w) << std::endl;
    int32_t o                   = a * w;
    int32_t oOriginal           = as[0] * ws[0] + as[1] * ws[1] + as[2] * ws[2];
    uint32_t oOriginalUnsinged  = (((uint)as[0]) & 0x3F) * (((uint)ws[0]) & 0x3F) + 
                                  (((uint)as[1]) & 0x3F) * (((uint)ws[1]) & 0x3F) + 
                                  (((uint)as[2]) & 0x3F) * (((uint)ws[2]) & 0x3F);
    std::string s = std::bitset<32>(o).to_string();
    size_t pos = s.find(std::bitset<7>(oOriginalUnsinged).to_string());
    std::cout << oOriginal << " -> 0b" << std::bitset<7>(oOriginal) << std::endl;
    std::cout << oOriginalUnsinged << " -> 0b" << std::bitset<7>(oOriginalUnsinged) << std::endl;
    std::cout << "0b" << std::bitset<16>(a) << " x 0b" << std::bitset<16>(w) << " = 0b" << std::bitset<32>(o) << std::endl;
    if (pos != std::string::npos){
        for(int i = 0 ; i < pos + 16 + 16 + 12 ; i++)
            std::cout << " ";
        for(int i = 0 ; i < 7   ; i++)
            std::cout << "^";
        std::cout << std::endl;
        std::cout << pos << " - " << pos + 7 << std::endl;
    }
    else{
        for (int i = 0 ; i < 32  ; i++)
            std::cout << "x";
        std::cout << std::endl;
    }
    std::cout << "0b" << std::bitset<16>(o) << std::endl;
    std::cout << ((o & (0x0000007f << (pos - 1))) >> (pos - 1)) << std::endl;
    return 0;
}