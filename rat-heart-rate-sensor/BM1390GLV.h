#ifndef BM1390GLV_H__
#define BM1390GLV_H__

#ifdef ARDUINO_AVR_UNO
#define float                             float
#endif

#define BM1390GLV_DEVICE_ADDRESS            (0x5D)

#define BM1390GLV_MANUFACTURER_ID_PART_ID   (0x0F)
#define BM1390GLV_POWER_DOWN                (0x12)
#define BM1390GLV_RESET                     (0x13)
#define BM1390GLV_MODE_CONTROL              (0x14)
#define BM1390GLV_PRESSURE_TEMPERATURE      (0x1A)

#define BM1390GLV_POWER_DOWN_PWR_DOWN       (1 << 0)
#define BM1390GLV_RESET_RSTB                (1 << 0)
#define BM1390GLV_MODE_CONTROL_AVE_NUM      (3 << 5)
#define BM1390GLV_MODE_CONTROL_MODE         (2 << 0)

#define BM1390GLV_MANUFACTURER_ID_VAL       (0xE0)
#define BM1390GLV_PART_ID_VAL               (0x34)
#define BM1390GLV_POWER_DOWN_VAL            (BM1390GLV_POWER_DOWN_PWR_DOWN)
#define BM1390GLV_RESET_VAL                 (BM1390GLV_RESET_RSTB)
#define BM1390GLV_MODE_CONTROL_VAL          (BM1390GLV_MODE_CONTROL_AVE_NUM)

#define BM1390GLV_DRIVER_VERSION            ("1.0")

#define BM1390GLV_MODE_MASK                 (0xFC)
#define BM1390GLV_MEASUREMENT               (BM1390GLV_MODE_CONTROL_MODE)
#define BM1390GLV_STANDBY                   (0x00)

#define BM1390GLV_MANUFACTURER_ID           (0)
#define BM1390GLV_PART_ID                   (1)
#define BM1390GLV_ALL_ID_SIZE               (2)
#define BM1390GLV_PRESS_OUT_MSB             (0)
#define BM1390GLV_PRESS_OUT_LSB             (1)
#define BM1390GLV_PRESS_OUT_XL              (2)
#define BM1390GLV_TEMP_OUT_MSB              (3)
#define BM1390GLV_TEMP_OUT_LSB              (4)
#define BM1390GLV_ALL_OUT_SIZE              (5)
#define BM1390GLV_COUNTS_PER_HPA            (8192)  // 4(2bit Shift) * 2048
#define BM1390GLV_COUNTS_PER_DEGREE_CELSIUS (32)

#define BM1390GLV_COMM_OK                   (0)
#define BM1390GLV_COMM_ERROR                (-1)
#define BM1390GLV_WAI_ERROR                 (-2)

class BM1390GLV
{
   public:
    BM1390GLV();
    ~BM1390GLV();
    String  get_driver_version(void);
    int32_t init(void);
    int32_t start(void);
    int32_t stop(void);
    int32_t get_val(uint8_t *raw);
    int32_t get_val(float *press, float *temp);

   private:
    uint8_t Address;
    uint8_t ModeCtlVal;

    int32_t init_check(void);
    int32_t init_setting(void);
    float convert_hpa(uint32_t raw_press);
    float convert_degree_celsius(int16_t raw_temp);
    int32_t write(uint8_t reg, uint8_t *data, int32_t size);
    int32_t read(uint8_t reg, uint8_t *data, int32_t size);
};

#endif // BM1390GLV_H__
