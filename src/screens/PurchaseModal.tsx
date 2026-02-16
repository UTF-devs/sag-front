import { X } from "lucide-react";
import React, { useState, useEffect } from "react";
import { toast, ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import { useLanguage } from "../contexts/LanguageContext";

interface PurchaseModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const PurchaseModal: React.FC<PurchaseModalProps> = ({ isOpen, onClose }) => {
  const { t } = useLanguage();

  const [form, setForm] = useState({
    full_name: "",
    phone_number: "",
    city: "",
    address: "",
    comment: "",
  });

  const cities = [
    t("Toshkent"),
    t("Toshkentvil"),
    t("Andijon"),
    t("Buxoro"),
    t("Fargona"),
    t("Jizzax"),
    t("Xorazm"),
    t("Namangan"),
    t("Navoiy"),
    t("Qashqadaryo"),
    t("Samarqand"),
    t("Sirdaryo"),
    t("Surxondaryo"),
  ];

  const [errors, setErrors] = useState({
    full_name: "",
    phone_number: "",
    city: "",
    address: "",
  });

  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }

    return () => {
      document.body.style.overflow = "";
    };
  }, [isOpen]);

  if (!isOpen) return null;

  const handleBackgroundClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  const validate = () => {
    const errs = { full_name: "", phone_number: "", city: "", address: "" };
    const nameRegex = /^[^\d]+$/;
    const phoneRegex = /^\+998\d{9}$/;

    if (!form.full_name.trim()) {
      errs.full_name = t("required_field") || "This field is required";
    } else if (!nameRegex.test(form.full_name)) {
      errs.full_name = t("name_invalid") || "Name cannot contain numbers";
    }

    if (!form.phone_number.trim()) {
      errs.phone_number = t("required_field") || "This field is required";
    } else if (!phoneRegex.test(form.phone_number)) {
      errs.phone_number = t("phone_invalid") || "Invalid phone number format";
    }

    if (!form.city.trim()) {
      errs.city = t("required_field") || "This field is required";
    }

    if (!form.address.trim()) {
      errs.address = t("required_field") || "This field is required";
    }

    setErrors(errs);
    return Object.values(errs).every((err) => err === "");
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!validate()) {
      toast.error(
        t("please_fill_required") || "Please fill all required fields",
      );
      return;
    }

    try {
      const successMessage = t("sent_successfully") || "Successfully sent!";
      toast.success(successMessage, {
        position: "top-right",
        autoClose: 3000,
        hideProgressBar: false,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
      });

      // Reset form
      setForm({
        full_name: "",
        phone_number: "",
        city: "",
        address: "",
        comment: "",
      });

      // Delay closing the modal to allow toast to display
      setTimeout(() => {
        onClose();
      }, 3500); // Adjust delay as needed (slightly longer than toast autoClose)
    } catch (err) {
      console.error("ERROR!", err);
      toast.error(t("error_occurred") || "An error occurred");
    }
  };

  const handleChange = (
    e: React.ChangeEvent<
      HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement
    >,
  ) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  return (
    <div
      onClick={handleBackgroundClick}
      className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-50"
    >
      <ToastContainer
        position="top-right"
        autoClose={3000}
        hideProgressBar={false}
        newestOnTop={false}
        closeOnClick
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
      />
      <div className="bg-[#FFFCE0] p-6 rounded-md w-full max-w-2xl relative shadow-lg">
        <button
          className="absolute top-3 right-3 text-gray-500 hover:text-gray-700"
          onClick={onClose}
        >
          <X size={20} />
        </button>

        <h2 className="text-xl font-semibold mb-4 text-center">{t("buy")}</h2>

        <form className="space-y-4" onSubmit={handleSubmit}>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">
                {t("name")}
              </label>
              <input
                type="text"
                name="full_name"
                value={form.full_name}
                onChange={handleChange}
                className={`w-full bg-[#FFFCE0] border p-2 rounded ${errors.full_name ? "border-red-500" : "border-gray-300"}`}
                placeholder={t("name")}
              />
              {errors.full_name && (
                <p className="text-red-500 text-sm mt-1">{errors.full_name}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700">
                {t("phone")}
              </label>
              <input
                type="tel"
                name="phone_number"
                value={form.phone_number}
                onChange={handleChange}
                className={`w-full bg-[#FFFCE0] border p-2 rounded ${errors.phone_number ? "border-red-500" : "border-gray-300"}`}
                placeholder="+998901234567"
              />
              {errors.phone_number && (
                <p className="text-red-500 text-sm mt-1">
                  {errors.phone_number}
                </p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700">
                {t("region")}
              </label>
              <select
                name="city"
                value={form.city}
                onChange={handleChange}
                className={`w-full bg-[#FFFCE0] border p-2 rounded ${errors.city ? "border-red-500" : "border-gray-300"}`}
              >
                <option value="">{t("select")}</option>
                {cities.map((city, idx) => (
                  <option key={idx} value={city}>
                    {city}
                  </option>
                ))}
              </select>
              {errors.city && (
                <p className="text-red-500 text-sm mt-1">{errors.city}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700">
                {t("address")}
              </label>
              <input
                type="text"
                name="address"
                value={form.address}
                onChange={handleChange}
                className={`w-full bg-[#FFFCE0] border p-2 rounded ${errors.address ? "border-red-500" : "border-gray-300"}`}
                placeholder={t("full_address")}
              />
              {errors.address && (
                <p className="text-red-500 text-sm mt-1">{errors.address}</p>
              )}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">
              {t("note")}
            </label>
            <textarea
              name="comment"
              value={form.comment}
              onChange={handleChange}
              className="w-full border bg-[#FFFCE0] border-gray-300 p-2 rounded"
              rows={3}
              placeholder={t("additional_notes")}
            />
          </div>

          <button
            type="submit"
            className="w-full bg-blue-900 hover:bg-blue-800 text-white py-2 rounded"
          >
            {t("send")}
          </button>
        </form>
      </div>
    </div>
  );
};

export default PurchaseModal;
