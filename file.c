#include <linux/module.h>
#include <linux/pci.h>
#include <linux/init.h>

#define DRIVER_NAME "my_pci_driver"

/* Supported PCI Devices - Vendor ID and Device ID */
static const struct pci_device_id pci_ids[] = {
    { PCI_DEVICE(0x8086, 0x100e) },  /* Example: Intel 82540EM Ethernet */
    { 0, }  /* Terminates the list */
};
MODULE_DEVICE_TABLE(pci, pci_ids);

/* Probe Function: Called when a matching PCI device is found */
static int my_pci_probe(struct pci_dev *pdev, const struct pci_device_id *ent)
{
    printk(KERN_INFO "PCI device found: Vendor=0x%X, Device=0x%X\n",
           pdev->vendor, pdev->device);
    
    if (pci_enable_device(pdev))
        return -EIO;

    return 0;
}

/* Remove Function: Called when the PCI device is removed */
static void my_pci_remove(struct pci_dev *pdev)
{
    printk(KERN_INFO "PCI device removed: Vendor=0x%X, Device=0x%X\n",
           pdev->vendor, pdev->device);
}

/* PCI Driver Structure */
static struct pci_driver my_pci_driver = {
    .name = DRIVER_NAME,
    .id_table = pci_ids,
    .probe = my_pci_probe,
    .remove = my_pci_remove,
};

/* Module Initialization */
static int __init my_pci_init(void)
{
    return pci_register_driver(&my_pci_driver);
}

/* Module Exit */
static void __exit my_pci_exit(void)
{
    pci_unregister_driver(&my_pci_driver);
}

module_init(my_pci_init);
module_exit(my_pci_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name");
MODULE_DESCRIPTION("A Simple PCI Driver with Hotplug Support");