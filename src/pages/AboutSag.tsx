import AboutSagSection from "../screens/AboutSagPage/AboutSag"
import Gallery from "../screens/AboutSagPage/Gallery"
import ProductionSection from "../screens/AboutSagPage/ProductionSection"
import { ContactInfoSection } from "../screens/HomePage/sections/ContactInfoSection"

const AboutSag = () => {
  return (
    <div className="bg-[#FFFCE0]">
         <AboutSagSection/>
         <ProductionSection/>
         <Gallery/>
         <ContactInfoSection/>
    </div>
  )
}

export default AboutSag