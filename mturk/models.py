import turkic.database
import turkic.models
from sqlalchemy import Column, Integer, String, Text, ForeignKey, Boolean, Table
from sqlalchemy.orm import relationship, backref


class Job(turkic.models.HIT):
    __tablename__ = "jobs"
    __mapper_args__ = {"polymorphic_identity": "jobs"}

    id = Column(Integer, ForeignKey(turkic.models.HIT.id), primary_key = True)
    category = Column(String(250))

    def getpage(self):
        return "?id={0}".format(self.id)

class DetectionWindow(turkic.database.Base):
    __tablename__ = "windows"

    id = Column(Integer, primary_key = True)
    filepath = Column(String(250))

    @property 
    def score(self):
        score = 0
        counter = 0
        for interconnect in self.interconnect:
            if interconnect.isgood is True:
                score += 1
                counter += 1
            if interconnect.isgood is False:
                counter += 1
        return score / float(counter)

class Interconnect(turkic.database.Base):
    __tablename__ = "interconnect"

    id = Column(Integer, primary_key = True)

    job_id = Column(Integer, ForeignKey(Job.id))
    job = relationship(Job, backref = backref("interconnect", cascade = "all,delete"))

    window_id = Column(Integer, ForeignKey(DetectionWindow.id))
    window = relationship(DetectionWindow, backref="interconnect", cascade = "all,delete")

    isgood = Column(Boolean)
